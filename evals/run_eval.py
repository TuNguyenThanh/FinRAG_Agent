"""Offline evaluation against a running FinRAG API.

    uv run python evals/run_eval.py --mode retrieval          # no LLM calls, free
    uv run python evals/run_eval.py --mode answer [--judge]   # end-to-end, costs tokens

Metrics
- retrieval: hit@1 / hit@5 (right document), evidence@5 (expected fact in a top-5 chunk), MRR
- answer:    accuracy (expected facts present), citation rate, abstention on out-of-scope
             questions, p50/p95 latency, tokens and cost per question, optional LLM-judge
             faithfulness (answer supported by retrieved sources).
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
ABSTAIN_MARKERS = ("không tìm thấy", "không có thông tin", "not find")


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


async def eval_retrieval(client: httpx.AsyncClient, rows: list[dict]) -> dict:
    rows = [r for r in rows if r["expected_doc"]]
    hit1 = hit5 = evidence = 0
    rr = []
    for r in rows:
        res = (await client.post("/v1/search", json={"query": r["question"], "top_k": 5})).json()
        docs = [x["doc_name"] for x in res["results"]]
        hit1 += docs[:1] == [r["expected_doc"]]
        hit5 += r["expected_doc"] in docs
        rank = next((i for i, d in enumerate(docs, 1) if d == r["expected_doc"]), None)
        rr.append(1 / rank if rank else 0.0)
        text = norm(" ".join(x["content"] for x in res["results"]))
        evidence += all(norm(m) in text for m in r["must_contain"])
    n = len(rows)
    return {
        "questions": n,
        "hit@1": round(hit1 / n, 3),
        "hit@5": round(hit5 / n, 3),
        "evidence@5": round(evidence / n, 3),
        "mrr": round(sum(rr) / n, 3),
    }


async def judge_faithfulness(judge, answer: str, sources: list[dict]) -> bool:
    context = "\n---\n".join(s["content"] for s in sources)
    prompt = (
        "You grade a bank assistant. Reply with only YES or NO.\n"
        "Is every factual claim (numbers, fees, rates, conditions) in the ANSWER supported "
        f"by the CONTEXT?\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    reply = await judge.ainvoke(prompt)
    return reply.text.strip().upper().startswith("YES")


async def eval_answers(
    client: httpx.AsyncClient, rows: list[dict], judge_model: str | None
) -> dict:
    judge = None
    if judge_model:
        from langchain.chat_models import init_chat_model

        judge = init_chat_model(judge_model, temperature=0)

    details, latencies, costs, tokens = [], [], [], []
    for r in rows:
        t0 = time.perf_counter()
        resp = await client.post(
            "/v1/chat",
            json={"message": r["question"], "thread_id": uuid.uuid4().hex, "stream": False},
            timeout=120,
        )
        data = resp.json()
        latencies.append((time.perf_counter() - t0) * 1000)
        costs.append(data.get("cost_usd", 0.0))
        tokens.append(data.get("input_tokens", 0) + data.get("output_tokens", 0))
        answer = norm(data.get("answer", ""))
        in_scope = r["expected_doc"] is not None
        row = {"question": r["question"], "answer": data.get("answer", ""), "in_scope": in_scope}
        if in_scope:
            row["correct"] = all(norm(m) in answer for m in r["must_contain"])
            row["cited"] = r["expected_doc"].removesuffix(".md") in answer
            if judge and data.get("sources"):
                row["faithful"] = await judge_faithfulness(judge, data["answer"], data["sources"])
        else:
            row["abstained"] = any(m in answer for m in ABSTAIN_MARKERS)
        details.append(row)

    scoped = [d for d in details if d["in_scope"]]
    oos = [d for d in details if not d["in_scope"]]
    summary = {
        "questions": len(details),
        "accuracy": round(sum(d["correct"] for d in scoped) / len(scoped), 3),
        "citation_rate": round(sum(d["cited"] for d in scoped) / len(scoped), 3),
        "abstention_rate_oos": round(sum(d["abstained"] for d in oos) / max(len(oos), 1), 3),
        "latency_p50_ms": round(statistics.median(latencies)),
        "latency_p95_ms": round(pct(latencies, 0.95)),
        "avg_tokens": round(statistics.mean(tokens)),
        "avg_cost_usd": round(statistics.mean(costs), 6),
    }
    judged = [d["faithful"] for d in scoped if "faithful" in d]
    if judged:
        summary["faithfulness"] = round(sum(judged) / len(judged), 3)
    return {"summary": summary, "details": details}


def print_table(title: str, summary: dict) -> None:
    print(f"\n### {title}\n\n| metric | value |\n|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v} |")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "answer", "all"], default="retrieval")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="dev-key")
    ap.add_argument("--dataset", default=str(ROOT / "dataset.jsonl"))
    ap.add_argument(
        "--judge",
        nargs="?",
        const="anthropic:claude-haiku-5-5",
        default=None,
        help="LLM-as-judge faithfulness (init_chat_model id)",
    )
    args = ap.parse_args()

    rows = load(Path(args.dataset))
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    async with httpx.AsyncClient(
        base_url=args.base_url, headers={"X-API-Key": args.api_key}, timeout=60
    ) as client:
        if args.mode in ("retrieval", "all"):
            res = await eval_retrieval(client, rows)
            print_table("Retrieval", res)
            (out_dir / "retrieval.json").write_text(json.dumps(res, indent=2))
        if args.mode in ("answer", "all"):
            res = await eval_answers(client, rows, args.judge)
            print_table("Answers", res["summary"])
            (out_dir / "answers.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
