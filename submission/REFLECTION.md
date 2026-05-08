# Lab 22 — DPO Alignment: Submission Reflection

**Student:** Nguyễn Văn Đạt (2A202600411)  
**Date:** 2026  
**Compute tier:** T4 (Colab free, Qwen2.5-3B-bnb-4bit)  
**GitHub repo:** https://github.com/datnv12333/2A202600411-NguyenVanDat-Day22-Track3-DPO-Alignment-Lab
**Hugging Face Hub (Option B):** https://huggingface.co/datnguyennn/day22-dpo-alignment

---

## § 1 — Tóm tắt pipeline

Pipeline Lab 22 được thực thi trên một Colab notebook duy nhất gồm 6 giai đoạn liên tiếp:

1. **SFT-mini (Stage 1):** Fine-tune `unsloth/Qwen2.5-3B-bnb-4bit` trên 1k mẫu từ `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` bằng LoRA `r=16`, `lora_alpha=32`, 1 epoch. Kết quả là adapter SFT tại `adapters/sft-mini/`.
2. **Preference data (Stage 2):** Load 2k cặp từ `argilla/ultrafeedback-binarized-preferences-cleaned`, format thành `{prompt, chosen, rejected}` theo Qwen2.5 chat template, và lưu Parquet tại `data/pref/train.parquet`.
3. **DPO training (Stage 3):** Train adapter DPO trên checkpoint SFT với `beta=0.1`, `lr=5e-7`, `1 epoch`. Notebook ghi nhận reward curves và lưu adapter tại `adapters/dpo/`.
4. **Compare & Eval (Stage 4):** Generate 8 prompts cố định (4 helpfulness + 4 safety) với cả hai adapter, render side-by-side table, rồi chạy judge theo rubric.
5. **Merge + GGUF (Stage 5):** Merge SFT+DPO sang FP16 để chuẩn bị GGUF. Tuy nhiên, trong bản notebook hiện tại, cell convert sang GGUF bị lỗi mapping tensor ở bước xuất F16 nên smoke test GGUF chưa hoàn tất.
6. **Benchmark (Stage 6):** Có code sẵn cho IFEval, GSM8K, MMLU, AlpacaEval-lite; kết quả cuối cùng cần được lấy từ output của NB6 sau khi chạy xong.

---

## § 2 — Số liệu thực tế

| Metric | Giá trị |
|---|---|
| GPU | Tesla T4 (15.6 GB) |
| COMPUTE_TIER | T4 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT final loss | 1.4693 |
| DPO final loss | 0.8300 |
| End chosen reward | -1.277 |
| End rejected reward | -1.525 |
| End reward gap | +0.249 |
| GGUF size | TBD / chưa có vì bước convert GGUF đang lỗi trong notebook hiện tại |
| IFEval SFT → DPO | TBD |
| GSM8K SFT → DPO | TBD |
| MMLU SFT → DPO | TBD |
| AlpacaEval-lite win-rate | TBD |

---

## § 3 — Phân tích reward curves (NB3)

Sau khi chạy DPO training với `beta=0.1` và learning rate `5e-7` trên 2,000 cặp UltraFeedback trong 1 epoch, reward curves trong notebook cho thấy đúng tín hiệu mà deck §3.4 muốn người học quan sát: `chosen_rewards`, `rejected_rewards`, và `reward gap = chosen - rejected`. Ở cuối run, `chosen_rewards` dừng ở khoảng `-1.277`, `rejected_rewards` ở khoảng `-1.525`, và gap là `+0.249`. Điều này quan trọng vì nó cho thấy policy đã đẩy response được chọn lên cao hơn so với reference, đồng thời kéo response bị loại xuống thấp hơn. Đây không phải chỉ là một con số đẹp; nó là dấu hiệu trực tiếp rằng DPO đang tối ưu đúng hướng.

Điểm đáng chú ý là các curve này vẫn dao động khá mạnh theo step, chứ không tăng đều như đường thẳng. Điều đó phù hợp với bối cảnh T4: chỉ 2k preference pairs, 1 epoch, batch hiệu dụng nhỏ, nên gradient signal rất nhiễu. Vì vậy, đọc reward curves phải nhìn cả xu hướng lẫn biên độ dao động. Nếu chỉ nhìn `reward gap` mà bỏ qua `chosen` và `rejected` riêng lẻ thì rất dễ diễn giải sai. Trường hợp trong notebook này không giống “likelihood displacement” kiểu chosen đi xuống còn rejected đi xuống nhanh hơn; thay vào đó, gap tăng đi kèm với chosen cải thiện và rejected suy giảm, tức là gần với pattern “intended” trong deck §3.4 hơn. Nói cách khác, run này cho thấy DPO đã tạo ra alignment tín hiệu có ý nghĩa, dù vẫn còn nhiễu và chưa đủ mạnh để kết luận về generalization ngoài tập train.

---

## § 4 — NB2: Preference data

Dataset `argilla/ultrafeedback-binarized-preferences-cleaned` được load thành 2,000 preference pairs và format theo chat template của Qwen2.5. Các thống kê trong notebook quan trọng hơn phần mô tả chung: median prompt length là `87`, median chosen length là `400`, median rejected length là `278`, và chỉ `44.2%` cặp fit trong `MAX_LEN=512`. Con số này cho thấy đây không phải một dataset “ngắn và sạch” như mình có thể nghĩ ban đầu; ngược lại, truncation là rủi ro có thật trong T4 tier.

Điều này ảnh hưởng trực tiếp đến chất lượng DPO. Vì chosen/rejected thường dài hơn prompt khá nhiều, một phần signal có thể bị cắt nếu max length quá thấp. Trong run hiện tại, notebook còn cảnh báo rằng tỷ lệ fit thấp hơn ngưỡng mong muốn 80%, nên việc dùng `MAX_LEN=512` là một trade-off giữa khả năng chạy được trên T4 và độ đầy đủ của training signal. Mặt khác, việc dùng UltraFeedback tiếng Anh thay vì preference data tiếng Việt native là lựa chọn có chủ đích để giữ pipeline chạy được trong phạm vi lab. Tóm lại, NB2 cho thấy dataset đủ lớn để DPO học được tín hiệu preference cơ bản, nhưng cũng đủ dài để tạo ra giới hạn thực tế về context budget.

---

## § 5 — NB4: Side-by-side comparison

Trong notebook, NB4 tạo 8 prompt cố định gồm 4 prompt helpfulness và 4 prompt safety, rồi so sánh output của SFT-only và SFT+DPO. Phần judge summary là điểm quan trọng nhất: kết quả hiện tại cho thấy `tie: 8/8` ở cả overall, helpfulness, và safety. Nghĩa là, theo rubric/judge đang dùng trong run này, hai adapter chưa tách nhau đủ mạnh để tạo ra một chiến thắng rõ ràng cho bên nào.

Đây là một kết quả đáng chú ý vì nó khác với kỳ vọng “DPO thắng rõ” mà nhiều người thường tự mặc định. Có vài cách đọc hợp lý: thứ nhất, 8 prompt là một sample quá nhỏ nên độ nhạy thống kê thấp; thứ hai, rubric/judge có thể chưa đủ tinh để nhận ra khác biệt; thứ ba, DPO đã học được thay đổi nhẹ về style nhưng chưa đủ mạnh để tạo ra thay đổi quan sát được trên tập prompt này. Với safety prompts, hiện tượng tie không có nghĩa là DPO vô dụng, mà thường có nghĩa là model đã rơi vào vùng trả lời tương đối an toàn ở cả hai cấu hình. Với helpfulness prompts, tie cũng cho thấy SFT-mini đã khá ổn trên các prompt được chọn. Vì vậy, phần NB4 trong reflection nên mô tả đây là một comparison “không phân tách rõ”, thay vì khẳng định DPO win-rate cao.

---

## § 6 — Quyết định thiết kế quan trọng nhất

Quyết định quan trọng nhất trong pipeline này là giữ nguyên **English UltraFeedback** làm preference dataset thay vì cố ép tạo một preference dataset tiếng Việt native ngay trong lab. Đây là quyết định ảnh hưởng đồng thời đến khả năng hoàn thành bài, tính so sánh với deck, và chất lượng diễn giải kết quả. Trong phạm vi T4 và thời lượng của một notebook lab, lựa chọn này hợp lý hơn so với việc tự build một bộ preference tiếng Việt từ đầu, vì dữ liệu native chất lượng cao gần như không có sẵn ở dạng public, còn quy trình tự gán nhãn sẽ tốn thời gian và công sức vượt scope.

Tác động kỹ thuật của quyết định này là cross-lingual gap. Model được tối ưu trên preference signal tiếng Anh, nhưng nhiều phần trong evaluation và smoke test lại dùng tiếng Việt. Điều đó có nghĩa là DPO có thể học được các quy luật chung như “trả lời có cấu trúc”, “đi thẳng vào câu hỏi”, hoặc “từ chối lịch sự khi unsafe”, nhưng không nhất thiết học được các nuance tiếng Việt như xưng hô, mức độ formal, hoặc kiểu từ chối tự nhiên theo văn hoá người dùng Việt. Đây là lý do tại sao một số benchmark thiên về format như IFEval có thể phù hợp với DPO, trong khi các kiểm tra thiên về ngôn ngữ tự nhiên hoặc safety phrasing tiếng Việt có thể không cải thiện mạnh. Ngoài ra, 2,000 cặp trong 1 epoch là cực nhỏ so với quy mô DPO production; vì vậy, run này chủ yếu chứng minh pipeline và tín hiệu học có tồn tại, hơn là chứng minh mô hình đã đạt mức “align” bền vững. Nếu phải chọn một thay đổi lớn nhất cho version sau, đó sẽ là thêm một lớp preference data tiếng Việt nhỏ nhưng native, thay vì chỉ dựa hoàn toàn vào English UltraFeedback.

---

## § 7 — Giải thích benchmark results và alignment tax

Notebook đã có đầy đủ code cho NB6: IFEval, GSM8K, MMLU và AlpacaEval-lite. Tuy nhiên, trong bản notebook hiện tại, các cell benchmark không mang theo output cuối cùng, nên chưa thể trích ra số delta chính xác để ghi thành kết luận định lượng. Vì vậy, phần này nên được đọc như khung diễn giải kết quả, còn các con số cuối cùng cần lấy từ `data/eval/benchmark_results.json` sau khi NB6 chạy xong.

Cách diễn giải đúng là như sau. IFEval là benchmark phù hợp nhất để bắt hiệu ứng DPO vì nó đo khả năng bám format và tuân theo instruction; nếu DPO có tác dụng, đây thường là chỗ tăng trước tiên. GSM8K là nơi dễ thấy alignment tax nhất: model càng được tune theo preference chatty thì càng có nguy cơ giảm nhẹ ở bài toán toán học, vì capacity bị kéo sang style và refusal handling thay vì reasoning sâu. MMLU thường ít đổi hơn vì DPO không tạo thêm factual knowledge mới; nó chủ yếu thay đổi cách trả lời. AlpacaEval-lite là benchmark gần với phân phối training nhất vì đều là judge-based preference style, nên nếu có cải thiện thì đó là dấu hiệu preference signal transfer tốt.

Điều quan trọng nhất là không diễn giải alignment như một thắng-thua tuyệt đối. Một run tốt có thể tăng IFEval và AlpacaEval-lite nhưng giữ MMLU gần như flat và chỉ giảm GSM8K một chút. Đó là profile đúng của alignment tax: không miễn phí, nhưng chấp nhận được nếu mục tiêu là chat quality. Khi có số cuối cùng, phần kết luận nên viết theo đúng delta thực tế thay vì gán sẵn một win-rate hay một mức tăng “mong đợi”.
