#!/usr/bin/env python
"""Interactive Web Server for Adaptive Minimax Connect Four Search.

Serves an interactive web application to visualize agent search decisions,
tactical features, VOC leaf evaluations, play games against AI, watch AI vs AI
battles, and inspect benchmark figures.

Usage:
    python serve.py                 # Starts server on http://127.0.0.1:8000
    python serve.py --port 8080     # Custom port
    python serve.py --no-browser    # Do not auto-open browser
"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import mimetypes
import socket
import os
from pathlib import Path
import sys
import threading
import time
import urllib.parse
import webbrowser

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from adaptive_minimax import config as C
from adaptive_minimax.evaluation import analyze, static_eval
from adaptive_minimax.experiments import load_artifacts, make_agents
from adaptive_minimax.features import FEATURE_NAMES, IDX, extract_features
from adaptive_minimax.game import HEIGHT, WIDTH, H1, State

WEB_DIR = ROOT_DIR / "web"
RESULTS_DIR = C.RESULT_DIR
FIG_DIR = C.FIG_DIR

_LOCK = threading.Lock()   # agents share caches/RNG -> serialise search calls

# Global preloaded artifacts & agents
print("Loading model artifacts and search agents...", flush=True)
_ART = load_artifacts()
_AGENTS = make_agents(_ART)
print("Artifacts loaded successfully. Ready to serve requests.", flush=True)


_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _safe_join(base: Path, rel: str) -> Path:
    """Join rel onto base but never escape base (blocks ../ path traversal)."""
    target = (base / urllib.parse.unquote(rel)).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        return base / "__not_found__"
    return target


def parse_moves(moves_input) -> list[int]:
    """Parse list of moves from integers, string, or comma-separated values."""
    if not moves_input:
        return []
    if isinstance(moves_input, str):
        if not moves_input.strip():
            return []
        parts = moves_input.replace(",", " ").split()
        return [int(x) for x in parts if x.isdigit()]
    if isinstance(moves_input, list):
        return [int(x) for x in moves_input]
    return []


def state_from_history(moves: list[int]) -> tuple[State, State | None]:
    """Reconstruct current state and parent state from a move history."""
    s = State()
    parent = None
    for m in moves:
        parent = s
        s = s.play(m)
    return s, parent


def serialize_board(state: State) -> dict:
    """Serialize the Connect 4 board into a JSON-friendly structure."""
    first_is_cur = (state.n % 2 == 0)
    p1 = state.cur if first_is_cur else state.opp  # Red (player 1)
    p2 = state.opp if first_is_cur else state.cur  # Yellow (player 2)

    grid = []
    for r in range(HEIGHT - 1, -1, -1):
        row = []
        for c in range(WIDTH):
            bit = 1 << (c * H1 + r)
            if p1 & bit:
                row.append(1)  # Red
            elif p2 & bit:
                row.append(2)  # Yellow
            else:
                row.append(0)  # Empty
        grid.append(row)

    winner = None
    if state.won:
        # Player who just moved won
        winner = 1 if (state.n % 2 == 1) else 2
    elif state.n >= HEIGHT * WIDTH:
        winner = 0  # Draw

    current_player = 1 if first_is_cur else 2

    return {
        "grid": grid,
        "ply": state.n,
        "is_terminal": state.is_terminal,
        "winner": winner,
        "current_player": current_player,
        "legal_moves": state.legal_moves(),
    }


def serialize_features(state: State, parent: State | None = None) -> dict:
    p_eval = static_eval(parent) if parent is not None else None
    feats = extract_features(state, p_eval)
    return {name: round(float(feats[idx]), 4) for idx, name in enumerate(FEATURE_NAMES)}


def introspect_agent(agent_name: str, state: State, parent: State | None, budget: int) -> dict:
    """Run agent and extract rich inspection metrics for the visualization."""
    agent = _AGENTS.get(agent_name)
    if agent is None:
        agent = _AGENTS["F"]
        agent_name = "F"

    with _LOCK:
        move, info = agent.choose(state, budget)

    p_eval = static_eval(parent) if parent is not None else None
    feats = extract_features(state, p_eval)

    details: dict = {
        "name": agent.name,
        "label": agent.label,
        "strategy": agent.__class__.__name__,
    }

    if agent_name == "A":
        depth = agent.depth_for(budget)
        details["calibrated_depth"] = depth
        details["description"] = f"Fixed-depth alpha-beta calibrated to average cost <= {budget} nodes."
    elif agent_name == "B":
        details["description"] = "Iterative deepening alpha-beta with partial-iteration reuse."
    elif agent_name == "C":
        swing_val = feats[IDX["swing"]]
        tactical_val = feats[IDX["my_win_now"]] + feats[IDX["opp_win_now"]]
        will_extend = tactical_val > 0 or swing_val >= agent.policy.swing_threshold
        details["swing_threshold"] = agent.policy.swing_threshold
        details["observed_swing"] = round(swing_val, 3)
        details["tactical_triggers"] = int(tactical_val)
        details["horizon_extended"] = will_extend
        details["description"] = "Selective extension if tactical threat exists or eval swing >= threshold."
    elif agent_name == "D":
        prob = float(agent.policy.predict_proba(feats))
        thr = float(agent.policy.threshold)
        details["ml_probability"] = round(prob, 4)
        details["threshold"] = round(thr, 4)
        details["horizon_extended"] = prob >= thr
        details["description"] = "Trained Gradient Boosting classifier decides whether extra search is worthwhile."
    elif agent_name in ("E", "F"):
        mu, sigma = agent.engine.model.predict(feats)
        details["mu"] = round(float(mu), 4)
        details["sigma"] = round(float(sigma), 4)
        details["k_root"] = agent.engine.k_root
        details["lam"] = agent.engine.lam
        details["model_kind"] = "Learned Gradient Boosting (F)" if agent_name == "F" else "Binned Empirical Table (E)"
        details["description"] = (
            "Best-first search allocating computation where Value-of-Computation "
            "(VOI from Monte-Carlo belief rollouts) is highest."
        )

    return {
        "move": move,
        "info": info,
        "details": details,
        "features": serialize_features(state, parent),
    }


class AdaptiveMinimaxHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path):
        if not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        # Explicit MIME types: on Windows the registry can map .css/.js to text/plain, which makes the
        # browser silently ignore the stylesheet/script (page opens but looks broken or does nothing).
        mime_type = _MIME.get(file_path.suffix.lower())
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html")
            return

        if path.startswith("/static/") or path.startswith("/web/"):
            rel_path = path.replace("/static/", "").replace("/web/", "")
            self._serve_file(_safe_join(WEB_DIR, rel_path))
            return

        if path.startswith("/figures/") or path.startswith("/results/figures/"):
            fname = Path(path).name
            self._serve_file(FIG_DIR / fname)
            return

        if path == "/api/state":
            qs = urllib.parse.parse_qs(parsed.query)
            moves_raw = qs.get("moves", [""])[0]
            moves = parse_moves(moves_raw)
            state, parent = state_from_history(moves)
            data = {
                "moves": moves,
                "board": serialize_board(state),
                "features": serialize_features(state, parent),
            }
            self._send_json(data)
            return

        if path == "/api/results":
            import pandas as pd
            res_data: dict = {"figures": []}
            if FIG_DIR.exists():
                res_data["figures"] = sorted([f.name for f in FIG_DIR.glob("*.png")])

            files_map = {
                "benchmark_summary": RESULTS_DIR / "benchmark_summary.csv",
                "tournament_summary": RESULTS_DIR / "tournament_summary.csv",
                "ablation": RESULTS_DIR / "ablation.csv",
                "classifier_comparison": RESULTS_DIR / "classifier_comparison.csv",
                "error_model_metrics": RESULTS_DIR / "error_model_metrics.csv",
            }
            for key, p in files_map.items():
                if p.exists():
                    try:
                        df = pd.read_csv(p)
                        res_data[key] = df.to_dict(orient="records")
                    except Exception as e:
                        res_data[key] = {"error": str(e)}

            tuning_file = RESULTS_DIR / "tuning.json"
            if tuning_file.exists():
                try:
                    res_data["tuning"] = json.loads(tuning_file.read_text())
                except Exception:
                    pass

            self._send_json(res_data)
            return

        # Fallback to web directory
        local_target = _safe_join(WEB_DIR, path.lstrip("/"))
        if local_target.is_file():
            self._serve_file(local_target)
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req = json.loads(post_body)
        except Exception:
            self._send_json({"error": "Invalid JSON in request body"}, 400)
            return

        if parsed.path == "/api/move":
            moves = parse_moves(req.get("moves", []))
            agent_name = req.get("agent", "F").upper()
            budget = int(req.get("budget", 1000))
            state, parent = state_from_history(moves)

            if state.is_terminal:
                self._send_json({
                    "error": "Position is already terminal",
                    "board": serialize_board(state),
                }, 400)
                return

            res = introspect_agent(agent_name, state, parent, budget)
            next_state = state.play(res["move"])
            res["board"] = serialize_board(state)
            res["next_board"] = serialize_board(next_state)
            res["moves"] = moves + [res["move"]]
            self._send_json(res)
            return

        if parsed.path == "/api/analyze":
            moves = parse_moves(req.get("moves", []))
            budget = int(req.get("budget", 300))
            state, parent = state_from_history(moves)

            if state.is_terminal:
                self._send_json({
                    "error": "Position is already terminal",
                    "board": serialize_board(state),
                }, 400)
                return

            results = {}
            for ag_key in ("A", "B", "C", "D", "E", "F"):
                results[ag_key] = introspect_agent(ag_key, state, parent, budget)

            consensus_moves = [r["move"] for r in results.values()]
            from collections import Counter
            counts = Counter(consensus_moves)
            most_common_move, top_count = counts.most_common(1)[0]

            self._send_json({
                "moves": moves,
                "board": serialize_board(state),
                "features": serialize_features(state, parent),
                "budget": budget,
                "consensus_move": most_common_move,
                "consensus_agreement": f"{top_count}/6 agents",
                "agents": results,
            })
            return

        self.send_error(404, "API endpoint not found")


def _start_ipv6_loopback(port):
    """Also listen on ::1 so http://localhost:PORT works when the OS resolves 'localhost' to IPv6."""
    if not socket.has_ipv6:
        return None

    class _V6Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    try:
        srv = _V6Server(("::1", port), AdaptiveMinimaxHandler)
    except OSError:
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run_server(port=8000, host="127.0.0.1", open_browser=True):
    server = None
    for p in range(port, port + 25):          # use the next free port if 8000 is busy
        try:
            server = ThreadingHTTPServer((host, p), AdaptiveMinimaxHandler)
            port = p
            break
        except OSError:
            print(f"Port {p} is busy, trying {p + 1} ...", flush=True)
    if server is None:
        raise SystemExit("Could not find a free port. Try: python serve.py --port 9000")
    server.daemon_threads = True
    v6 = _start_ipv6_loopback(port) if host in ("127.0.0.1", "localhost") else None
    url = f"http://localhost:{port}"
    print("\n=======================================================", flush=True)
    print(" Adaptive Minimax Interactive Web Dashboard", flush=True)
    print(f" Open in your browser:  {url}", flush=True)
    print(f"                   or:  http://127.0.0.1:{port}", flush=True)
    print(" (do NOT open web/index.html directly - it must be served by this script)", flush=True)
    print(" Press Ctrl+C to stop the server", flush=True)
    print("=======================================================\n", flush=True)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
        if v6 is not None:
            v6.shutdown()
            v6.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host, open_browser=not args.no_browser)
