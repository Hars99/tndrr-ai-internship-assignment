# tndrr AI Internship Assignment

Three experiments on fine-tuning a small language model, retrieving evidence from a technical book, and running two model variants locally.

[Assignment report (PDF)](tndrr_assignment_report.pdf)

| Task | What I tested | Result |
| --- | --- | --- |
| [1. LoRA fine-tuning](Task%201%20Finetune/) | Qwen2.5-0.5B-Instruct classifying mock interview requests into six intents | Exact-match accuracy rose from **21/60 (35%)** to **42/60 (70%)** |
| [2. PDF retrieval and RAG](Task%202-Rag/) | 80-token and 200-token chunks on 15 questions from *Pro Git* | Recall@5 rose from **5/15 (33.3%)** to **10/15 (66.7%)** |
| [3. Local inference benchmark](Task%203-local%20model/) | FP16 and Q4_K_M GGUF variants of Qwen2.5-0.5B-Instruct in LM Studio | Mean decode speed was **86.1 vs 140.5 tokens/s**. Warmed VRAM above idle was **1,105 vs 519 MiB** |

## Task 1 · LoRA fine-tuning

The synthetic dataset contains 120 training messages and 60 held-out test messages. The baseline and fine-tuned model used the same prompt and greedy decoding. I trained LoRA adapters on `q_proj` and `v_proj` for three epochs with rank 8 and alpha 16. The run updated 540,672 parameters. Fine-tuning corrected 23 baseline mistakes and introduced two new ones.

To reproduce the run, open the [notebook](Task%201%20Finetune/tndrr_task1_lora.ipynb) in Google Colab, choose a GPU runtime, and upload the [dataset](Task%201%20Finetune/task1-lora/tndrr_task1_intents.jsonl) when prompted. Run the cells from top to bottom.

[Baseline predictions](Task%201%20Finetune/task1-lora/base_predictions.jsonl) · [LoRA predictions](Task%201%20Finetune/task1-lora/lora_predictions.jsonl)

## Task 2 · PDF retrieval and RAG

The script extracts text page by page from the 501-page PDF, creates overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and ranks them by cosine similarity using NumPy. Both chunk sizes use a 20-token overlap. A Recall@5 hit requires a retrieved chunk on the annotated PDF page to contain the full evidence passage.

From the repository root, run these commands in a Python environment:

```powershell
cd "Task 2-Rag"
python -m pip install numpy pypdf sentence-transformers
python rag.py eval progit.pdf questions.json
```

The retrieval evaluation runs without LM Studio. To rerun answer generation, start the LM Studio server on port 1234 and load the Q4_K_M model with the identifier `tndrr-q4`. Then run:

```powershell
python rag.py demo retrieval_results.json --size 200 --id 2
```

Q02 retrieved the right page first and generated the SHA-1 answer without a page citation. For Q08, the right passage also ranked first, but the model returned `--S` instead of `-S`.

[Source book](https://git-scm.com/book/en/v2) · [Evaluation questions](Task%202-Rag/questions.json) · [Retrieval results](Task%202-Rag/retrieval_results.json) · [Q02 demo](Task%202-Rag/rag_demo.json) · [Q08 demo](Task%202-Rag/rag_demo_q08.json)

## Task 3 · Local inference benchmark

I ran the same 10 prompts on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM. Both runs used a 2,048-token context, temperature 0, and a 192-token output cap. Throughput is the mean decode rate across the matched prompts. VRAM is the difference between the warmed and idle GPU readings from `nvidia-smi`. Both models made instruction-following and reasoning errors on this prompt set.

From the repository root, start the LM Studio server and run the benchmark. Download or import both GGUF variants into LM Studio first. If your model keys differ, use the matching entries from `lms ls`.

```powershell
cd "Task 3-local model"
lms server start --port 1234
python benchmark.py --fp16-key "qwen25-05b@fp16" --q4-key "qwen25-05b@q4_k_m"
```

[Benchmark results](Task%203-local%20model/benchmark_results.json) · [Prompt outputs](Task%203-local%20model/benchmark_outputs.md)
