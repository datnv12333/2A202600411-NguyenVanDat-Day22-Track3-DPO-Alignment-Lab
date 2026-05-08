# Screenshots — Lab 22 DPO Alignment

Thư mục này chứa tối thiểu 7 screenshots sau khi chạy notebook:

| File | Nội dung | Stage |
|---|---|---|
| `01-setup-gpu.png` | nvidia-smi output: GPU name, VRAM | Setup |
| `02-sft-loss.png` | Loss curve SFT-mini (step vs loss, giảm đều) | NB1 |
| `03-dpo-reward-curves.png` | chosen_rewards + rejected_rewards + gap | NB3 |
| `04-side-by-side-table.png` | 8 prompts × SFT vs DPO outputs | NB4 |
| `05-judge-output.png` | Judge verdict hoặc manual rubric | NB4 |
| `06-gguf-smoke.png` | llama-cpp-python smoke test output tiếng Việt | NB5 |
| `07-benchmark-comparison.png` | 4-bar chart IFEval/GSM8K/MMLU/AlpacaEval Δ | NB6 |

## Yêu cầu

- Crop rõ ràng, đọc được text
- Không để lộ API key trong ảnh
- Screenshot 06 phải thấy filename GGUF load line

## Ghi chú

Các ảnh này được tạo tự động bởi notebook (matplotlib savefig).
Sau khi chạy xong, copy từ `submission/screenshots/` trên Colab về máy local.
