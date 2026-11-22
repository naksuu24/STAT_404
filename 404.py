#!/usr/bin/env python3
"""
STAT 404 project: Search algorithm x maze structure on pathfinding efficiency.

Design (matches proposal)
  Block (generation method G): dfs (recursive backtracking), prim, kruskal
  Whole-plot factors: size S in {20, 50, 100}, loop density D in {5, 15, 30}%
  Replicates: 10 mazes per G x S x D  -> 270 mazes
  Subplot factor: algorithm A in {bfs, dijkstra, astar, greedy, random_walk}
  Every maze is solved by all 5 algorithms -> 1350 rows

Randomization
  - each maze gets its own seed drawn from a master seed (recorded in CSV)
  - run order of the 270 mazes is shuffled
  - order of the 5 algorithms is shuffled per maze (recorded as alg_order)

Loop density definition (documented choice, change in add_loops if you prefer):
  A generator makes a perfect maze (a spanning tree, no loops).  We then knock
  down D% of the *remaining* interior walls (walls between two adjacent cells
  that are still closed), which creates D% extra passages = loops.

Usage
  python run_experiment.py                       # full experiment
  python run_experiment.py --reps 1 --out test.csv   # quick smoke test
"""
import argparse, csv, gc, heapq, random, time
from collections import deque

# ----------------------------------------------------------------------------
# Maze representation: cells are ints r*n+c; adj[i] = set of connected cells.
# Start = top-left (0), End = bottom-right (n*n-1).
# ----------------------------------------------------------------------------
def all_edges(n):
    edges = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            if c + 1 < n: edges.append((i, i + 1))
            if r + 1 < n: edges.append((i, i + n))
    return edges

def grid_neighbors(n, i):
    r, c = divmod(i, n)
    if r > 0: yield i - n
    if r < n - 1: yield i + n
    if c > 0: yield i - 1
    if c < n - 1: yield i + 1

def carve(adj, a, b):
    adj[a].add(b); adj[b].add(a)

# ---- generators (all produce a perfect maze / spanning tree) ---------------
def gen_dfs(n, rng):
    """Randomized depth-first search (recursive backtracker), iterative."""
    adj = [set() for _ in range(n * n)]
    seen = [False] * (n * n)
    start = rng.randrange(n * n)
    seen[start] = True
    stack = [start]
    while stack:
        cur = stack[-1]
        opts = [j for j in grid_neighbors(n, cur) if not seen[j]]
        if not opts:
            stack.pop(); continue
        nxt = rng.choice(opts)
        seen[nxt] = True
        carve(adj, cur, nxt)
        stack.append(nxt)
    return adj

def gen_prim(n, rng):
    """Randomized Prim: random frontier edge that touches the tree."""
    adj = [set() for _ in range(n * n)]
    in_tree = [False] * (n * n)
    start = rng.randrange(n * n)
    in_tree[start] = True
    frontier = [(start, j) for j in grid_neighbors(n, start)]
    while frontier:
        k = rng.randrange(len(frontier))
        frontier[k], frontier[-1] = frontier[-1], frontier[k]
        a, b = frontier.pop()
        if in_tree[b]:
            continue
        in_tree[b] = True
        carve(adj, a, b)
        frontier.extend((b, j) for j in grid_neighbors(n, b) if not in_tree[j])
    return adj

def gen_kruskal(n, rng):
    """Randomized Kruskal with union-find."""
    adj = [set() for _ in range(n * n)]
    parent = list(range(n * n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = all_edges(n)
    rng.shuffle(edges)
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            carve(adj, a, b)
    return adj

GENERATORS = {"dfs": gen_dfs, "prim": gen_prim, "kruskal": gen_kruskal}

def add_loops(adj, n, density, rng):
    """Open `density` fraction of the still-closed interior walls."""
    closed = [(a, b) for a, b in all_edges(n) if b not in adj[a]]
    k = round(density * len(closed))
    for a, b in rng.sample(closed, k):
        carve(adj, a, b)
    return k

# ----------------------------------------------------------------------------
# Solvers.  Each returns dict(success, nodes_explored, path_length, steps).
#   nodes_explored = nodes expanded (popped/visited) before the goal is reached
#   (random walk: number of distinct cells visited; steps = total moves)
# ----------------------------------------------------------------------------
def _path_len(parent, goal):
    L = 0
    while parent[goal] != -1:
        goal = parent[goal]; L += 1
    return L

def manhattan(n, i, goal):
    r1, c1 = divmod(i, n); r2, c2 = divmod(goal, n)
    return abs(r1 - r2) + abs(c1 - c2)

def solve_bfs(adj, n, s, g, rng=None, **_):
    parent = {s: -1}
    q = deque([s]); expanded = 0
    while q:
        u = q.popleft(); expanded += 1
        if u == g:
            return dict(success=1, nodes_explored=expanded,
                        path_length=_path_len(parent, g), steps=expanded)
        for v in adj[u]:
            if v not in parent:
                parent[v] = u; q.append(v)
    return dict(success=0, nodes_explored=expanded, path_length=-1, steps=expanded)

def solve_dijkstra(adj, n, s, g, rng=None, **_):
    dist = {s: 0}; parent = {s: -1}; done = set()
    pq = [(0, s)]; expanded = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in done: continue
        done.add(u); expanded += 1
        if u == g:
            return dict(success=1, nodes_explored=expanded,
                        path_length=_path_len(parent, g), steps=expanded)
        for v in adj[u]:
            nd = d + 1
            if nd < dist.get(v, 1 << 60):
                dist[v] = nd; parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dict(success=0, nodes_explored=expanded, path_length=-1, steps=expanded)

def solve_astar(adj, n, s, g, rng=None, **_):
    gscore = {s: 0}; parent = {s: -1}; done = set()
    pq = [(manhattan(n, s, g), 0, s)]; expanded = 0
    while pq:
        _, d, u = heapq.heappop(pq)
        if u in done: continue
        done.add(u); expanded += 1
        if u == g:
            return dict(success=1, nodes_explored=expanded,
                        path_length=_path_len(parent, g), steps=expanded)
        for v in adj[u]:
            nd = d + 1
            if nd < gscore.get(v, 1 << 60):
                gscore[v] = nd; parent[v] = u
                heapq.heappush(pq, (nd + manhattan(n, v, g), nd, v))
    return dict(success=0, nodes_explored=expanded, path_length=-1, steps=expanded)

def solve_greedy(adj, n, s, g, rng=None, **_):
    parent = {s: -1}; seen = {s}
    pq = [(manhattan(n, s, g), s)]; expanded = 0
    while pq:
        _, u = heapq.heappop(pq); expanded += 1
        if u == g:
            return dict(success=1, nodes_explored=expanded,
                        path_length=_path_len(parent, g), steps=expanded)
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parent[v] = u
                heapq.heappush(pq, (manhattan(n, v, g), v))
    return dict(success=0, nodes_explored=expanded, path_length=-1, steps=expanded)

def solve_random_walk(adj, n, s, g, rng=None, max_steps=10**6, **_):
    """Uniform random walk; stops at goal or after max_steps (censored)."""
    nbrs = [tuple(a) for a in adj]          # tuples for fast rng.choice
    cur = s; visited = {s}; steps = 0
    choice = rng.choice
    while cur != g and steps < max_steps:
        cur = choice(nbrs[cur]); steps += 1
        visited.add(cur)
    ok = int(cur == g)
    return dict(success=ok, nodes_explored=len(visited),
                path_length=steps if ok else -1, steps=steps)

SOLVERS = {"bfs": solve_bfs, "dijkstra": solve_dijkstra, "astar": solve_astar,
           "greedy": solve_greedy, "random_walk": solve_random_walk}

# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------
SIZES = [20, 50, 100]
DENSITIES = [0.05, 0.15, 0.30]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10, help="mazes per G x S x D cell")
    ap.add_argument("--seed", type=int, default=404, help="master seed")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--rw-cap-factor", type=int, default=200,
                    help="random-walk step cap = factor * n^2 (censored if hit)")
    ap.add_argument("--timing-repeats", type=int, default=3,
                    help="repeat each (deterministic) solve and keep the median time")
    args = ap.parse_args()

    master = random.Random(args.seed)

    # 1. Build the plan: one entry per maze, each with its own recorded seed.
    plan = []
    for gen in GENERATORS:
        for n in SIZES:
            for d in DENSITIES:
                for rep in range(1, args.reps + 1):
                    plan.append(dict(gen=gen, size=n, density=d, rep=rep,
                                     seed=master.getrandbits(32)))
    master.shuffle(plan)                       # randomize maze run order
    for i, m in enumerate(plan, 1):
        m["maze_id"] = i                       # id = position in run order

    # 2. Pre-generate all mazes (outside the timed portion).
    print(f"Generating {len(plan)} mazes ...")
    for m in plan:
        rng = random.Random(m["seed"])
        t0 = time.perf_counter()
        adj = GENERATORS[m["gen"]](m["size"], rng)
        m["loops_added"] = add_loops(adj, m["size"], m["density"], rng)
        m["gen_time"] = time.perf_counter() - t0
        m["adj"] = adj
        m["opt_len"] = solve_bfs(adj, m["size"], 0, m["size"] ** 2 - 1)["path_length"]

    # 3. Run every algorithm on every maze, in random algorithm order.
    fields = ["maze_id", "seed", "gen_method", "size", "loop_density", "rep",
              "loops_added", "opt_path_length", "alg_order", "run_position",
              "algorithm", "time_s", "nodes_explored", "steps", "success",
              "path_length", "gen_time_s"]
    gc_was = gc.isenabled()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for m in plan:
            n = m["size"]; s, g = 0, n * n - 1
            order = list(SOLVERS); master.shuffle(order)
            for pos, name in enumerate(order, 1):
                solver = SOLVERS[name]
                reps = 1 if name == "random_walk" else args.timing_repeats
                times = []
                gc.collect(); gc.disable()
                for _ in range(reps):
                    # random walk gets a fresh seeded RNG so it is reproducible
                    rng = random.Random(m["seed"] * 31 + pos)
                    t0 = time.perf_counter()
                    res = solver(m["adj"], n, s, g, rng=rng,
                                 max_steps=args.rw_cap_factor * n * n)
                    times.append(time.perf_counter() - t0)
                if gc_was: gc.enable()
                times.sort()
                w.writerow(dict(
                    maze_id=m["maze_id"], seed=m["seed"], gen_method=m["gen"],
                    size=n, loop_density=m["density"], rep=m["rep"],
                    loops_added=m["loops_added"], opt_path_length=m["opt_len"],
                    alg_order="-".join(order), run_position=pos, algorithm=name,
                    time_s=times[len(times) // 2], nodes_explored=res["nodes_explored"],
                    steps=res["steps"], success=res["success"],
                    path_length=res["path_length"], gen_time_s=m["gen_time"]))
            f.flush()
            if m["maze_id"] % 10 == 0:
                print(f"  maze {m['maze_id']}/{len(plan)} done")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()