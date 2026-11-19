import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from pypdf import PdfReader

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ANSWER_MODEL = "tndrr-q4"
CHUNK_SIZES = (80, 200)
OVERLAP = 20
TOP_K = 5


def clean(text):
    return " ".join(text.split())


def read_pages(pdf_path):
    reader = PdfReader(pdf_path)
    if len(reader.pages) < 30:
        raise ValueError("The assignment requires a public source of at least 30 pages.")
    return [clean(page.extract_text() or "") for page in reader.pages]


def load_questions(path, pages):
    questions = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(questions, list) or len(questions) != 15:
        raise ValueError("questions.json must contain 15 questions.")
    seen = set()
    for i, row in enumerate(questions, 1):
        row["question"] = clean(row["question"])
        row["passage"] = clean(row["passage"])
        page = row["page"]
        if not row["question"] or not row["passage"] or type(page) is not int:
            raise ValueError(f"Question {i}: fill in the question, PDF page, and passage.")
        if row["question"].casefold() in seen:
            raise ValueError(f"Question {i}: duplicate question.")
        seen.add(row["question"].casefold())
        if not 1 <= page <= len(pages) or row["passage"] not in pages[page - 1]:
            raise ValueError(f"Question {i}: copy the exact passage from PDF page {page}.")
    return questions


def tokenize_pages(pages, tokenizer):
    if not tokenizer.is_fast:
        raise ValueError("This implementation needs the embedding model's fast tokenizer.")
    return [
        tokenizer(page, add_special_tokens=False, truncation=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        for page in pages
    ]


def make_chunks(pages, page_offsets, size):
    chunks = []
    for page_number, (page, offsets) in enumerate(zip(pages, page_offsets), 1):
        start = 0
        while start < len(offsets):
            end = min(start + size, len(offsets))
            text = page[offsets[start][0]:offsets[end - 1][1]]
            chunks.append({"page": page_number, "text": text})
            if end == len(offsets):
                break
            start = end - OVERLAP
    if not chunks:
        raise ValueError("No searchable text found. Use the text-based Pro Git PDF.")
    return chunks


def embed(model, texts, show_progress=False):
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )


def ranks(query_vector, index):
    # Dot product equals cosine similarity because embeddings are normalized.
    scores = index @ query_vector
    return np.argsort(-scores, kind="stable"), scores


def evaluate(model, pages, page_offsets, questions):
    query_vectors = embed(model, [row["question"] for row in questions])
    runs = []
    for size in CHUNK_SIZES:
        chunks = make_chunks(pages, page_offsets, size)
        index = embed(model, [chunk["text"] for chunk in chunks], show_progress=True)
        rows = []
        for gold, vector in zip(questions, query_vectors):
            order, scores = ranks(vector, index)
            first_relevant_rank = next(
                (
                    rank
                    for rank, idx in enumerate(order, 1)
                    if chunks[idx]["page"] == gold["page"]
                    and gold["passage"] in chunks[idx]["text"]
                ),
                None,
            )
            rows.append({
                **gold,
                "relevant_rank": first_relevant_rank,
                "hit_at_5": first_relevant_rank is not None and first_relevant_rank <= TOP_K,
                "top_5": [
                    {**chunks[idx], "cosine_similarity": float(scores[idx])}
                    for idx in order[:TOP_K]
                ],
            })
        hits = sum(row["hit_at_5"] for row in rows)
        print(f"{size} tokens: {len(chunks)} chunks, Recall@5 {hits}/{len(rows)} ({hits / len(rows):.1%})")
        for i, row in enumerate(rows, 1):
            if not row["hit_at_5"]:
                rank = row["relevant_rank"]
                reason = f"gold chunk ranked {rank}" if rank else "no chunk contains the full passage"
                print(f"  Q{i:02d}: {reason} | {row['question']}")
        runs.append({
            "chunk_tokens": size, "chunks": len(chunks), "recall_at_5": hits / len(rows),
            "hits_at_5": hits, "questions": rows,
        })
    return runs


def answer_question(question, passages, model):
    context = "\n\n".join(f"[PDF page {chunk['page']}] {chunk['text']}" for chunk in passages)
    payload = {
        "model": model,
        "system_prompt": (
            "Answer only from the source passages. Cite supporting PDF page numbers. "
            "If evidence is insufficient, say so. Keep the answer brief."
        ),
        "input": f"Source passages:\n{context}\n\nQuestion: {question}",
        "stream": False,
        "store": False,
        "temperature": 0,
        "max_output_tokens": 180,
    }
    headers = {"Content-Type": "application/json"}
    if token := os.getenv("LM_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        "http://127.0.0.1:1234/api/v1/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urlopen(request, timeout=120) as response:
            outputs = json.load(response)["output"]
            return "\n".join(item["content"] for item in outputs if item["type"] == "message")
    except HTTPError as error:
        raise ValueError(f"LM Studio HTTP {error.code}: {error.read().decode()}") from error
    except URLError as error:
        raise ValueError("Start the LM Studio local server on port 1234 and load an answer model.") from error


def print_answer(question, passages, model):
    print("Retrieved PDF pages:", ", ".join(str(chunk["page"]) for chunk in passages))
    answer = answer_question(question, passages, model)
    if not answer.strip():
        raise ValueError("LM Studio returned no answer text.")
    print(answer)
    return {"question": question, "model": model, "passages": passages, "answer": answer}


def main():
    parser = argparse.ArgumentParser(description="Build and evaluate semantic search over Pro Git.")
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source", help="Find or display a PDF page.")
    source.add_argument("pdf", type=Path)
    lookup = source.add_mutually_exclusive_group(required=True)
    lookup.add_argument("--find")
    lookup.add_argument("--page", type=int)
    score = commands.add_parser("eval", help="Evaluate two chunk sizes on 15 questions.")
    score.add_argument("pdf", type=Path)
    score.add_argument("questions", type=Path)
    score.add_argument("--out", type=Path, default=Path("retrieval_results.json"))
    ask = commands.add_parser("ask", help="Retrieve source passages and ask the local LLM.")
    ask.add_argument("pdf", type=Path)
    ask.add_argument("question")
    ask.add_argument("--size", choices=CHUNK_SIZES, type=int, required=True)
    ask.add_argument("--model", default=ANSWER_MODEL)
    demo = commands.add_parser("demo", help="Generate an answer from saved evaluation retrieval.")
    demo.add_argument("results", type=Path)
    demo.add_argument("--size", choices=CHUNK_SIZES, type=int, required=True)
    demo.add_argument("--model", default=ANSWER_MODEL)
    demo.add_argument("--id", type=int, required=True, help="Question number, 1 to 15.")
    args = parser.parse_args()

    if args.command == "demo":
        results = json.loads(args.results.read_text(encoding="utf-8"))
        run = next(run for run in results["runs"] if run["chunk_tokens"] == args.size)
        if not 1 <= args.id <= len(run["questions"]):
            raise ValueError("Question number is outside the saved evaluation.")
        row = run["questions"][args.id - 1]
        answer = print_answer(row["question"], row["top_5"], args.model)
        Path("rag_demo.json").write_text(json.dumps(answer, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    pages = read_pages(args.pdf)
    print(f"PDF pages: {len(pages)}")
    if args.command == "source":
        if args.page is not None:
            if not 1 <= args.page <= len(pages):
                raise ValueError("Use the PDF viewer's page position, starting at 1.")
            print(pages[args.page - 1])
        else:
            query = clean(args.find).casefold()
            if not query:
                raise ValueError("Enter a phrase to find.")
            for i, page in enumerate(pages, 1):
                match = page.casefold().find(query)
                if match >= 0:
                    print(f"PDF page {i}: {page[max(0, match - 55):match + len(query) + 110]}")
        return

    if args.command == "eval":
        questions = load_questions(args.questions, pages)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, device="cpu")
    page_offsets = tokenize_pages(pages, model.tokenizer)
    if args.command == "eval":
        result = {
            "source": args.pdf.name,
            "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            "pdf_pages": len(pages),
            "embedding_model": MODEL_ID,
            "overlap_tokens": OVERLAP,
            "top_k": TOP_K,
            "runs": evaluate(model, pages, page_offsets, questions),
        }
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {args.out}")
    else:
        chunks = make_chunks(pages, page_offsets, args.size)
        index = embed(model, [chunk["text"] for chunk in chunks], show_progress=True)
        order, _ = ranks(embed(model, [args.question])[0], index)
        print_answer(args.question, [chunks[idx] for idx in order[:TOP_K]], args.model)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from None

