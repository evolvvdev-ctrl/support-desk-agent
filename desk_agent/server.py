"""Small HTTP wrapper so an evaluation tool can drive the agent.

POST /chat    {"prompt": "..."}  ->  {"output", "tool_calls", "steps", "stop_reason"}
GET  /state   refunds, escalations, outbox and the full tool-event log
POST /reset   restore the seed data and clear all logs
GET  /health
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import DeskAgent
from .llm import OpenAICompatibleLLM
from .tools import Sandbox

MAX_BODY = 64 * 1024


def make_handler(agent: DeskAgent) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one conversation at a time: the sandbox is shared state

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "guards": agent.sandbox.guards})
            elif self.path == "/state":
                with lock:
                    self._send(200, agent.sandbox.state())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/reset":
                with lock:
                    agent.sandbox.reset()
                self._send(200, {"ok": True})
                return
            if self.path != "/chat":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._send(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                prompt = body["prompt"]
                if not isinstance(prompt, str):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                self._send(400, {"error": 'expected JSON {"prompt": "<text>"}'})
                return
            with lock:
                try:
                    result = agent.run(prompt)
                except Exception as exc:  # model backend down, bad reply shape, etc.
                    self._send(502, {"error": f"model backend error: {exc}"})
                    return
            self._send(200, {
                "output": result.output,
                "tool_calls": result.tool_calls,
                "steps": result.steps,
                "stop_reason": result.stop_reason,
            })

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Support desk agent HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--guards", choices=["on", "off"], default=os.getenv("DESK_GUARDS", "on"))
    args = parser.parse_args()
    agent = DeskAgent(OpenAICompatibleLLM(), Sandbox.from_data_dir(guards=args.guards == "on"))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent))
    print(f"desk-agent on http://{args.host}:{args.port}  guards={args.guards}  model={agent.llm.model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
