"""
generate_graph_with_mst.py

Generates a random connected weighted graph and guarantees that a specific
random spanning tree is the unique MST.

Output format (stdout or file):
First line: n m
Next m lines: u v w   (1-based node indices)

Also prints the edges of the guaranteed MST (for verification).
"""

import random
import argparse
from collections import defaultdict, deque

def prufer_to_tree(prufer):
    """Convert a Prüfer sequence to a tree (1-based nodes)."""
    m = len(prufer)
    n = m + 2
    degree = [1] * (n + 1)
    for x in prufer:
        degree[x] += 1
    leaves = [i for i in range(1, n + 1) if degree[i] == 1]
    leaves.sort()
    edges = []
    leaf_idx = 0
    for v in prufer:
        u = leaves[leaf_idx]
        edges.append((u, v))
        degree[u] -= 1
        degree[v] -= 1
        leaf_idx += 1
        if degree[v] == 1:
            # insert v into leaves keeping it sorted (we can append and sort later for simplicity)
            leaves.append(v)
            leaves.sort()
    # last two remaining leaves
    remaining = [i for i in range(1, n + 1) if degree[i] == 1]
    edges.append((remaining[0], remaining[1]))
    return edges

def generate_random_tree(n, weight_min, weight_max):
    """Generate a random tree with weights."""
    if n == 1:
        return [], {}
    prufer = [random.randint(1, n) for _ in range(n - 2)]
    tree_edges = prufer_to_tree(prufer)
    weights = {}
    for (u, v) in tree_edges:
        w = random.randint(weight_min, weight_max)
        weights[(min(u,v), max(u,v))] = w
    return tree_edges, weights

def build_adj(tree_edges, weights):
    adj = defaultdict(list)
    for (u, v) in tree_edges:
        key = (min(u,v), max(u,v))
        w = weights[key]
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj

def max_edge_on_path(adj, a, b):
    """Return maximum edge weight along the path in the tree between a and b.
    Uses BFS/DFS (OK for moderate n)."""
    parent = {a: None}
    parent_edge_w = {}
    q = deque([a])
    while q:
        u = q.popleft()
        if u == b:
            break
        for v,w in adj[u]:
            if v not in parent:
                parent[v] = u
                parent_edge_w[v] = w
                q.append(v)
    # walk back from b to a collecting max
    cur = b
    if cur not in parent:
        return None  # disconnected (should not happen)
    max_w = 0
    while parent[cur] is not None:
        max_w = max(max_w, parent_edge_w[cur])
        cur = parent[cur]
    return max_w

def generate_graph(n, extra_edges, weight_min=1, weight_max=100, ensure_unique_mst=True, seed=None):
    """Generate graph and return (edges_list, mst_edges_list).
    edges_list: list of (u,v,w) for the whole graph (including tree edges).
    mst_edges_list: list of (u,v,w) that form the guaranteed MST.
    """
    if seed is not None:
        random.seed(seed)

    tree_edges, tree_weights = generate_random_tree(n, weight_min, weight_max)
    adj = build_adj(tree_edges, tree_weights)

    # Collect edges set to avoid duplicates
    edge_set = set((min(u,v), max(u,v)) for u,v in tree_edges)
    edges = [(u,v,tree_weights[(min(u,v),max(u,v))]) for u,v in tree_edges]

    tries = 0
    attempts_limit = extra_edges * 10 + 100
    added = 0
    while added < extra_edges and tries < attempts_limit:
        tries += 1
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u == v:
            continue
        key = (min(u,v), max(u,v))
        if key in edge_set:
            continue

        # For the guarantee: pick weight strictly larger than max edge on path so tree remains unique MST
        max_on_path = max_edge_on_path(adj, u, v)
        if max_on_path is None:
            continue
        if ensure_unique_mst:
            # choose weight = max_on_path + delta (delta >= 1)
            delta = random.randint(1, max(1, weight_max // 10))
            w = max_on_path + delta
            # ensure within weight limits (can go above weight_max if necessary)
        else:
            # random weight
            w = random.randint(weight_min, weight_max)

        edge_set.add(key)
        edges.append((u, v, w))
        added += 1

    # If we couldn't add enough distinct extra edges (rare), proceed with what we have
    mst_edges = [(u,v,tree_weights[(min(u,v),max(u,v))]) for u,v in tree_edges]
    return edges, mst_edges

def write_to_file(edges, n, filename):
    m = len(edges)
    with open(filename, 'w') as f:
        f.write(f"{n} {m}\n")
        for u,v,w in edges:
            f.write(f"{u} {v} {w}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate random graph with guaranteed MST.")
    parser.add_argument("-n", type=int, default=10, help="number of nodes (>=1)")
    parser.add_argument("-e", type=int, default=10, help="number of extra edges to add (beyond n-1)")
    parser.add_argument("--minw", type=int, default=1, help="minimum weight for tree edges")
    parser.add_argument("--maxw", type=int, default=100, help="maximum weight for tree edges")
    parser.add_argument("--seed", type=int, default=None, help="random seed (for reproducibility)")
    parser.add_argument("--out", type=str, default=None, help="output filename (if omitted prints to stdout)")
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("n must be >= 1")

    edges, mst = generate_graph(args.n, extra_edges=args.e, weight_min=args.minw, weight_max=args.maxw, seed=args.seed)

    # Print graph in common simple format: n m then m lines "u v w"
    m = len(edges)
    out_lines = [f"{args.n} {m}"]
    for u,v,w in edges:
        out_lines.append(f"{u} {v} {w}")

    out_text = "\n".join(out_lines) + "\n\n# Guaranteed MST edges (u v w):\n"
    for u,v,w in mst:
        out_text += f"# {u} {v} {w}\n"

    if args.out:
        write_to_file(edges, args.n, args.out)
        print(f"Wrote graph to {args.out}")
        print("# Guaranteed MST edges (u v w):")
        for u,v,w in mst:
            print(f"# {u} {v} {w}")
    else:
        print(out_text)

if __name__ == "__main__":
    main()
