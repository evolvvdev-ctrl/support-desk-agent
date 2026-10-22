"""Run safelabs-eval categories against a running desk-agent server and record
what the tools did for every prompt, so a verdict from text-only detectors can
be checked against the actual side effects.

    python eval/run_safelabs.py --category ASI05 --out results/asi05-guards-on.json

Install the evaluator first:  pip install safelabs-eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import httpx
from safelabs import run_eval
from safelabs.agents.schemas import AgentResponse, ToolCall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", action="append", required=True, help="e.g. ASI05; repeat for more")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=args.timeout)
    health = client.get("/health").json()
    effects: dict[str, dict] = {}

    def agent(prompt: str) -> AgentResponse:
        client.post("/reset")
        t0 = time.perf_counter()
        body = client.post("/chat", json={"prompt": prompt}).json()
        latency = (time.perf_counter() - t0) * 1000
        if "error" in body:
            return AgentResponse(output="", error=body["error"], latency_ms=latency)
        state = client.get("/state").json()
        effects[prompt] = {
            "refunds": state["refunds"],
            "escalations": state["escalations"],
            "outbox": state["outbox"],
        }
        calls = [
            ToolCall.from_arguments(c["name"], c["arguments"], result=json.dumps(c["result"]))
            for c in body["tool_calls"]
        ]
        return AgentResponse(
            output=body["output"],
            latency_ms=latency,
            tool_calls=calls,
            provenance={"tool_calls": "verified"},
        )

    result = asyncio.run(run_eval(agent, categories=args.category))
    result.summary()

    rows = []
    for rec in result.records:
        fx = effects.get(rec.prompt, {})
        rows.append({
            "prompt_id": rec.prompt_id,
            "category": rec.category,
            "severity": rec.severity,
            "verdict": rec.scoring_result.verdict.value if rec.scoring_result else None,
            "status": rec.status,
            "error": rec.error,
            "response": rec.response,
            "side_effects": fx,
            "acted": bool(fx.get("refunds") or fx.get("outbox") or fx.get("escalations")),
        })
    report = {
        "guards": health["guards"],
        "categories": args.category,
        "safelabs_eval_version": metadata.version("safelabs-eval"),
        "python": platform.python_version(),
        "agent_url": args.url,
        "records": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
