#!/usr/bin/env python3
"""
Geodesic Memory Client SDK & CLI 3.0
Provides zero-boilerplate access to the Geodesic Memory 3.0 MCP Server.
"""

import sys
import os
import json
import subprocess

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

class GeodesicMemoryClient:
    def __init__(self, server_path: str = SERVER_PATH):
        self.server_path = server_path
        self._proc = None
        self._msg_id = 0
        self._start_server()

    def _start_server(self):
        cmd = [sys.executable, "-u", self.server_path]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self._send_rpc("initialize")

    def _send_rpc(self, method: str, params: dict = None) -> dict:
        self._msg_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": method
        }
        if params is not None:
            req["params"] = params

        raw = json.dumps(req)
        frame = f"Content-Length: {len(raw.encode('utf-8'))}\r\n\r\n{raw}".encode("utf-8")
        self._proc.stdin.write(frame)
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline().decode("utf-8")
            if not line:
                raise RuntimeError("MCP Server closed stdout unexpectedly.")
            if line.lower().startswith("content-length:"):
                length = int(line.split(":")[1].strip())
                while True:
                    sep = self._proc.stdout.readline().decode("utf-8")
                    if sep.strip() == "":
                        break
                body = self._proc.stdout.read(length).decode("utf-8")
                res = json.loads(body)
                if "error" in res:
                    raise RuntimeError(f"RPC Error: {res['error']}")
                return res

    def _call_tool(self, tool_name: str, args: dict) -> dict:
        res = self._send_rpc("tools/call", {"name": tool_name, "arguments": args})
        content_text = res["result"]["content"][0]["text"]
        try:
            return json.loads(content_text)
        except Exception:
            return {"raw_output": content_text}

    def query(self, text: str, top_k: int = 5, min_resonance: float = 0.20, resolution: str = "all", temporal_octave: int = None) -> dict:
        """Fast wave-interference query with dual-spectral rotor and dyadic resolution filtering."""
        args = {
            "query": text,
            "top_k": top_k,
            "min_resonance": min_resonance,
            "resolution": resolution
        }
        if temporal_octave is not None:
            args["temporal_octave"] = temporal_octave
        return self._call_tool("memory_superposition_query", args)

    def traverse(self, concept: str, max_depth: int = 3, direction: str = "forward", resolution: str = "meso") -> dict:
        """Follows relativistic frame-dragging angular momentum across causal chains."""
        return self._call_tool("memory_geodesic_traverse", {
            "start_concept": concept,
            "max_depth": max_depth,
            "direction": direction,
            "resolution": resolution
        })

    def predict(self, start_concept: str, steps: int = 15, dt: float = 0.08) -> dict:
        """Forecasts next research breakthrough via symplectic Verlet phase-flow integration."""
        return self._call_tool("memory_trajectory_predict", {
            "start_concept": start_concept,
            "steps": steps,
            "dt": dt
        })

    def sleep(self, tau_causal_days: float = 7.0, ltp_rate: float = 0.06, ltd_rate: float = 0.03, hard_threshold: float = 0.05) -> dict:
        """Executes biological STDP Hebbian sleep replay consolidation and parsimony pruning."""
        return self._call_tool("memory_sleep_consolidate", {
            "tau_causal_days": tau_causal_days,
            "ltp_rate": ltp_rate,
            "ltd_rate": ltd_rate,
            "hard_threshold": hard_threshold
        })

    def latch(self, concept: str, content: str = "", macro: str = None, meso: str = None, micro: str = None,
              category: str = "general", latch_strength: float = 4.208, links: list = None) -> dict:
        """Permanently latches a concept into phase space with supercritical pitchfork strength."""
        args = {
            "concept": concept,
            "content": content,
            "category": category,
            "latch_strength": latch_strength,
            "links": links or []
        }
        if macro: args["macro"] = macro
        if meso: args["meso"] = meso
        if micro: args["micro"] = micro
        return self._call_tool("memory_latch", args)

    def entangle(self, min_resonance: float = 0.32, max_edges_per_node: int = 6) -> dict:
        """Triggers autonomous wave-interference entanglement across the entire manifold."""
        return self._call_tool("memory_entangle_manifold", {
            "min_resonance": min_resonance,
            "max_edges_per_node": max_edges_per_node
        })

    def read_full(self, concept_or_id: str, file_name: str = None, start_line: int = 1, max_lines: int = 300) -> dict:
        """Holographic Source Dereferencer: pulls full source code, papers, or transcripts from disk."""
        args = {
            "concept_or_id": concept_or_id,
            "start_line": start_line,
            "max_lines": max_lines
        }
        if file_name:
            args["file_name"] = file_name
        return self._call_tool("memory_read_full", args)

    def inspect(self) -> dict:
        """Inspects continuous manifold topology, Hamiltonian energy, and active latch states."""
        return self._call_tool("memory_inspect_manifold", {})

    def close(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Geodesic Memory 3.0 CLI Client")
    subparsers = parser.add_subparsers(dest="command")

    # Query command
    q_parser = subparsers.add_parser("query")
    q_parser.add_argument("text", help="Query cue")
    q_parser.add_argument("--top-k", type=int, default=3)
    q_parser.add_argument("--res", choices=["macro", "meso", "micro", "all"], default="meso")
    q_parser.add_argument("--min-res", type=float, default=0.18)
    q_parser.add_argument("--octave", type=int, default=None)

    # Traverse command
    t_parser = subparsers.add_parser("traverse")
    t_parser.add_argument("concept", help="Origin concept")
    t_parser.add_argument("--depth", type=int, default=3)
    t_parser.add_argument("--direction", choices=["forward", "backward", "bidirectional"], default="forward")
    t_parser.add_argument("--res", choices=["macro", "meso", "micro"], default="meso")

    # Predict command
    pr_parser = subparsers.add_parser("predict")
    pr_parser.add_argument("concept", help="Starting concept to extrapolate from")
    pr_parser.add_argument("--steps", type=int, default=15)
    pr_parser.add_argument("--dt", type=float, default=0.08)

    # Sleep command
    s_parser = subparsers.add_parser("sleep")
    s_parser.add_argument("--tau", type=float, default=7.0)
    s_parser.add_argument("--ltp", type=float, default=0.06)
    s_parser.add_argument("--ltd", type=float, default=0.03)

    # Entangle command
    e_parser = subparsers.add_parser("entangle")
    e_parser.add_argument("--min-resonance", type=float, default=0.32)
    e_parser.add_argument("--max-edges", type=int, default=6)

    # Read Full command
    r_parser = subparsers.add_parser("read")
    r_parser.add_argument("concept", help="Concept title or node ID to dereference")
    r_parser.add_argument("--file", help="Specific file name inside suite/session", default=None)
    r_parser.add_argument("--start", type=int, default=1)
    r_parser.add_argument("--lines", type=int, default=200)

    # Inspect command
    subparsers.add_parser("inspect")

    args = parser.parse_args()

    client = GeodesicMemoryClient()
    try:
        if args.command == "query":
            res = client.query(args.text, top_k=args.top_k, min_resonance=args.min_res, resolution=args.res, temporal_octave=args.octave)
            print(json.dumps(res, indent=2))
        elif args.command == "read":
            res = client.read_full(args.concept, file_name=args.file, start_line=args.start, max_lines=args.lines)
            print(json.dumps(res, indent=2))
        elif args.command == "traverse":
            res = client.traverse(args.concept, max_depth=args.depth, direction=args.direction, resolution=args.res)
            print(json.dumps(res, indent=2))
        elif args.command == "predict":
            res = client.predict(args.concept, steps=args.steps, dt=args.dt)
            print(json.dumps(res, indent=2))
        elif args.command == "sleep":
            res = client.sleep(tau_causal_days=args.tau, ltp_rate=args.ltp, ltd_rate=args.ltd)
            print(json.dumps(res, indent=2))
        elif args.command == "entangle":
            res = client.entangle(min_resonance=args.min_resonance, max_edges_per_node=args.max_edges)
            print(json.dumps(res, indent=2))
        elif args.command == "inspect":
            res = client.inspect()
            print(json.dumps(res, indent=2))
        else:
            parser.print_help()
    finally:
        client.close()

if __name__ == "__main__":
    main()
