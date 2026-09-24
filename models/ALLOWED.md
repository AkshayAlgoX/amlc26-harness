# Approved pretrained weights

2025 rule (re-check the 2026 PDF on Day 1): final model(s) must be MIT or Apache-2.0 licensed and ≤ 8B parameters.
Code may only download weights listed here with status **OK**. A human opens the model card, checks licence + size,
fills the row, and flips the status. Candidates below are pre-filled but UNVERIFIED — check every one.

| HF id | params | revision (pinned) | licence (from model card) | status | checked by | use |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-VL-8B-Instruct | 8.77B (8,767,123,696) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 (verify) | TODO | | VLM: extraction / VQA / LoRA fine-tune |
| Qwen/Qwen3.5-2B | ~2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Apache-2.0 (verify) | TODO | | fast VLM baseline |
| PaddlePaddle/PaddleOCR-VL | ~0.9B | `7fa00a8c55b735ba51ba49a9058f3f9c57a99a11` | verify | TODO | | OCR |
| google/siglip2-so400m-patch14-384 | 1.14B (1,136,008,498; full checkpoint incl. text tower) | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | verify | TODO | | image encoder |
| openai/clip-vit-large-patch14 | ~0.4B | `32bd64288804d66eefd0ccbe215aa642df71cc41` | verify | TODO | | image-text encoder |
| microsoft/deberta-v3-base | ~0.2B | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` | verify | TODO | | text encoder |
| answerdotai/ModernBERT-base | ~0.15B | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | verify | TODO | | text encoder (long context) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | verify | TODO | | text embeddings |

Not allowed unless the 2026 rules change: anything under custom licences (e.g. Gemma terms, which covers
EmbeddingGemma), anything > 8B (e.g. Qwen3.5-9B), and any checkpoint fine-tuned on Amazon/product/price data.
