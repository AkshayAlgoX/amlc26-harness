# Approved pretrained weights

2025 rule (re-check the 2026 PDF on Day 1): final model(s) must be MIT or Apache-2.0 licensed and ≤ 8B parameters.
Code may only download weights listed here with status **OK**. A human opens the model card, checks licence + size,
fills the row, and flips the status. Candidates below are pre-filled but UNVERIFIED — check every one.

| HF id | params | licence (from model card) | status | checked by | use |
|---|---|---|---|---|---|
| Qwen/Qwen3-VL-8B-Instruct | ~8B | Apache-2.0 (verify) | TODO | | VLM: extraction / VQA / LoRA fine-tune |
| Qwen/Qwen3.5-2B | ~2B | Apache-2.0 (verify) | TODO | | fast VLM baseline |
| PaddlePaddle/PaddleOCR-VL | ~0.9B | verify | TODO | | OCR |
| google/siglip2-so400m-patch14-384 | ~0.4B | verify | TODO | | image encoder |
| openai/clip-vit-large-patch14 | ~0.4B | verify | TODO | | image-text encoder |
| microsoft/deberta-v3-base | ~0.2B | verify | TODO | | text encoder |
| answerdotai/ModernBERT-base | ~0.15B | verify | TODO | | text encoder (long context) |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | verify | TODO | | text embeddings |

Not allowed unless the 2026 rules change: anything under custom licences (e.g. Gemma terms, which covers
EmbeddingGemma), anything > 8B (e.g. Qwen3.5-9B), and any checkpoint fine-tuned on Amazon/product/price data.
