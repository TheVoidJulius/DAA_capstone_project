"""
Mini-Courier Planner — Dispatch Console UI
------------------------------------------
A zero-dependency web UI (Python standard library only) that drives the
real algorithms in courier_planner.py:

    Merge Sort  ->  0/1 Knapsack DP  ->  Dijkstra  ->  Nearest-Neighbor TSP

Run:   python ui_server.py
Open:  http://localhost:8000
"""

import ast
import io
import json
import sys
import time
import types
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# courier_planner imports matplotlib at the top for its PNG chart. The UI draws
# its own chart, so if matplotlib isn't installed we provide a harmless stand-in.
try:
    import matplotlib.pyplot  # noqa: F401
except ImportError:
    _mpl = types.ModuleType("matplotlib")
    _mpl.pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _mpl.pyplot

from courier_planner import (  # noqa: E402
    merge_sort_parcels,
    knapsack_dp_select,
    dijkstra_shortest_paths,
    plan_tsp_nearest_neighbor,
)
from datasets import get_small_dataset, generate_scaled_dataset  # noqa: E402

HUB = "Warehouse"
MAX_CAPACITY = 500
MAX_PARCELS = 60


def _run_knapsack_with_table(sorted_parcels, capacity):
    """Run the project's knapsack and capture the DP table it prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        selected, max_val, used_w = knapsack_dp_select(sorted_parcels, capacity, show_table=True)
    table = [ast.literal_eval(line.strip()) for line in buf.getvalue().splitlines()
             if line.strip().startswith("[") and not line.strip().startswith("[Phase")]
    return selected, max_val, used_w, table


def _backtrack_cells(table, parcels, capacity):
    """Cells visited while backtracking through the DP table (for highlighting)."""
    cells, w = [], capacity
    for i in range(len(parcels), 0, -1):
        took = table[i][w] != table[i - 1][w]
        cells.append({"i": i, "w": w, "took": took})
        if took:
            w -= parcels[i - 1]["weight"]
    cells.append({"i": 0, "w": w, "took": False})
    return cells


def _validate(payload, graph):
    try:
        capacity = int(payload["capacity"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Van capacity must be a whole number.")
    if not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError(f"Van capacity must be between 1 and {MAX_CAPACITY} kg.")

    raw = payload.get("parcels") or []
    if not isinstance(raw, list) or not raw:
        raise ValueError("Add at least one parcel.")
    if len(raw) > MAX_PARCELS:
        raise ValueError(f"At most {MAX_PARCELS} parcels.")

    parcels, seen = [], set()
    for n, p in enumerate(raw, 1):
        pid = str(p.get("id", "")).strip() or f"P{n}"
        if pid in seen:
            raise ValueError(f"Duplicate parcel ID '{pid}'.")
        seen.add(pid)
        dest = p.get("dest")
        if dest not in graph or dest == HUB:
            raise ValueError(f"{pid}: unknown destination '{dest}'.")
        try:
            weight, value, deadline = int(p["weight"]), int(p["value"]), int(p["deadline"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{pid}: weight, value and deadline must be whole numbers.")
        if weight < 1 or value < 0 or deadline < 0:
            raise ValueError(f"{pid}: weight must be ≥ 1, value and deadline ≥ 0.")
        parcels.append({"id": pid[:20], "dest": dest, "weight": weight,
                        "value": value, "deadline": deadline})
    return capacity, parcels


def plan(payload):
    _, _, graph = get_small_dataset()
    capacity, parcels = _validate(payload, graph)
    timings = {}

    t = time.perf_counter()
    sorted_p = merge_sort_parcels(parcels)
    timings["sort"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    selected, max_val, used_w, table = _run_knapsack_with_table(sorted_p, capacity)
    timings["knapsack"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    hub_dist, hub_prev = dijkstra_shortest_paths(graph, HUB)
    timings["dijkstra"] = (time.perf_counter() - t) * 1000

    route = None
    if selected:
        t = time.perf_counter()
        tour, road, km, _ = plan_tsp_nearest_neighbor(graph, HUB, [p["dest"] for p in selected])
        timings["tsp"] = (time.perf_counter() - t) * 1000
        route = {"tour": tour, "road": road, "km": km}

    return {
        "capacity": capacity,
        "sorted": sorted_p,
        "selected_ids": [p["id"] for p in selected],
        "max_value": max_val,
        "used_weight": used_w,
        "dp_table": table,
        "backtrack": _backtrack_cells(table, sorted_p, capacity),
        "hub_dist": hub_dist,
        "hub_prev": hub_prev,
        "route": route,
        "timings_ms": timings,
    }


def benchmark():
    sizes = [10, 50, 100, 250, 500, 1000]
    rows = []
    for n in sizes:
        with contextlib.redirect_stdout(io.StringIO()):
            cap, parcels, graph = generate_scaled_dataset(num_parcels=n, num_nodes=30, capacity=150)
        t0 = time.perf_counter()
        sorted_p = merge_sort_parcels(parcels)
        t1 = time.perf_counter()
        selected, _, _ = knapsack_dp_select(sorted_p, cap, show_table=False)
        t2 = time.perf_counter()
        plan_tsp_nearest_neighbor(graph, HUB, [p["dest"] for p in selected])
        t3 = time.perf_counter()
        rows.append({"n": n, "sort": (t1 - t0) * 1000, "knapsack": (t2 - t1) * 1000,
                     "route": (t3 - t2) * 1000})
    return {"rows": rows}


def dataset():
    capacity, parcels, graph = get_small_dataset()
    return {"capacity": capacity, "parcels": parcels, "graph": graph, "hub": HUB}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "ui" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/dataset":
            self._send(200, dataset())
        elif self.path == "/api/benchmark":
            self._send(200, benchmark())
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/plan":
            return self._send(404, {"error": "Not found"})
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 200_000)
            self._send(200, plan(json.loads(self.rfile.read(length) or b"{}")))
        except ValueError as e:
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Mini-Courier Dispatch Console running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
