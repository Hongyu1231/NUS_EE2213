#!/usr/bin/env python3
"""
Test runner for A2_A0311920U

Usage:
  python test_A2_runner.py                # run all built-in tests
  python test_A2_runner.py --case grid    # run a specific case (tiny|branch|undirected|nopth|grid)
  python test_A2_runner.py --module your_submission.py  # path to your file if named differently
"""

import importlib.util
import argparse
import sys
import time
import math
import random
from typing import Dict, Tuple, List

# ---------- Helpers ----------

def load_submission(module_path: str):
    spec = importlib.util.spec_from_file_location("submission", module_path)
    if spec is None:
        print(f"Could not load module from {module_path}")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "A2_A0311920U"):
        print("Module does not define function A2_A0311920U")
        sys.exit(1)
    return mod.A2_A0311920U

def path_cost(road_map: Dict[str, Dict[str, float]], path: List[str]) -> float:
    if not path or len(path) == 1:
        return 0.0
    cost = 0.0
    for u, v in zip(path, path[1:]):
        if u not in road_map or v not in road_map[u]:
            return math.inf
        cost += road_map[u][v]
    return cost

def check_coords_for_all(road_map, coords):
    missing = [c for c in road_map.keys() if c not in coords]
    return missing

# ---------- Test graphs ----------

def tiny_graph():
    # Simple line A->B->C
    road_map = {'A': {'B': 1}, 'B': {'C': 2}, 'C': {}}
    coords = {'A': (0,0), 'B': (1,0), 'C': (2,0)}
    start, goal = 'A', 'C'
    expect_cost = 3.0
    return road_map, coords, start, goal, expect_cost

def branch_graph():
    # A -> B (10), A -> C (1), C -> D (1), D -> B (1) so shortest A->C->D->B cost = 3
    road_map = {
        'A': {'B': 10, 'C': 1},
        'B': {'E': 2},
        'C': {'D': 1},
        'D': {'B': 1, 'E': 100},
        'E': {}
    }
    coords = {
        'A': (0,0), 'B': (5,0), 'C': (1,0), 'D': (2,0), 'E': (6,0)
    }
    start, goal = 'A', 'B'
    expect_cost = 3.0
    return road_map, coords, start, goal, expect_cost

def undirected_graph():
    # Symmetric edges; multiple shortest paths with same cost
    edges = {
        ('A','B',2), ('B','C',2), ('A','C',4), ('C','D',1), ('B','D',3)
    }
    road_map = {'A':{}, 'B':{}, 'C':{}, 'D':{}}
    for u,v,w in edges:
        road_map[u][v] = w
        road_map[v][u] = w
    coords = {'A': (0,0), 'B': (1,0), 'C': (2,0), 'D': (3,0)}
    start, goal = 'A', 'D'
    expect_cost = 5.0  # A-B (2) + B-D (3) or A-C (4) + C-D (1)
    return road_map, coords, start, goal, expect_cost

def no_path_graph():
    road_map = {'A': {'B': 1}, 'B': {}, 'C': {}}
    coords = {'A': (0,0), 'B': (1,0), 'C': (2,0)}
    start, goal = 'A', 'C'
    expect_cost = math.inf
    return road_map, coords, start, goal, expect_cost

def grid_graph(n=5, m=5, weight=1.0, diag=False, seed=0):
    # Build an n x m grid; nodes named like "r_c"
    random.seed(seed)
    road_map = {}
    coords = {}
    for r in range(n):
        for c in range(m):
            name = f"{r}_{c}"
            coords[name] = (c, r)  # x=c, y=r
            road_map[name] = {}
    # neighbors (4-dir or 8-dir if diag)
    dirs4 = [(1,0), (-1,0), (0,1), (0,-1)]
    dirs8 = dirs4 + [(1,1), (1,-1), (-1,1), (-1,-1)]
    dirs = dirs8 if diag else dirs4
    for r in range(n):
        for c in range(m):
            u = f"{r}_{c}"
            for dx,dy in dirs:
                nr, nc = r + dy, c + dx
                if 0 <= nr < n and 0 <= nc < m:
                    v = f"{nr}_{nc}"
                    # add slight randomization to weights to avoid too many ties
                    w = weight + (random.random()*0.1)
                    road_map[u][v] = w
    start, goal = "0_0", f"{n-1}_{m-1}"
    # expected cost is unknown due to randomized weights; we'll only sanity-check
    return road_map, coords, start, goal, None

# ---------- Runner ----------

def run_case(func, name, gen):
    print(f"\n=== Case: {name} ===")
    road_map, coords, s, t, expect = gen()
    missing = check_coords_for_all(road_map, coords)
    if missing:
        print(f"[!] Missing coordinates for: {missing}")
        return False

    t0 = time.time()
    path, cost = func(road_map, coords, s, t)
    dt = (time.time() - t0) * 1000.0
    print(f"start={s} goal={t}")
    print(f"path={path}")
    print(f"cost={cost}")
    print(f"time={dt:.2f} ms")

    if path is None:
        if expect is None or math.isinf(expect):
            print("[OK] correctly reported no path.")
            return True
        else:
            print("[FAIL] path is None but expected cost", expect)
            return False

    # Validate cost matches the edges
    recomputed = path_cost(road_map, path)
    if abs(recomputed - cost) > 1e-6:
        print(f"[FAIL] reported cost {cost} != recomputed {recomputed}")
        return False

    if expect is not None and not math.isinf(expect):
        if abs(cost - expect) > 1e-6:
            print(f"[FAIL] expected {expect} but got {cost}")
            return False

    print("[OK]")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="A2_A0311920U.py",
                    help="path to your submission module (default: A2_A0311920U.py)")
    ap.add_argument("--case", choices=["tiny","branch","undirected","nopth","grid"],
                    help="run only one specific case")
    ap.add_argument("--grid", type=int, default=6,
                    help="grid size N for the grid case (NxN)")
    args = ap.parse_args()

    func = load_submission(args.module)

    cases = {
        "tiny": tiny_graph,
        "branch": branch_graph,
        "undirected": undirected_graph,
        "nopth": no_path_graph,
        "grid": (lambda: grid_graph(args.grid, args.grid))
    }

    if args.case:
        ok = run_case(func, args.case, cases[args.case])
        sys.exit(0 if ok else 1)
    else:
        all_ok = True
        for name, gen in cases.items():
            ok = run_case(func, name, gen)
            all_ok &= ok
        print("\nALL PASS" if all_ok else "\nSOME TESTS FAILED")
        sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
