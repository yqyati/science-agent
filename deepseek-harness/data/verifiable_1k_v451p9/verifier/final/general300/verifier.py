"""Exact, self-contained evaluators for 25 manually specified training families.

The public task carries only the instance; no reference construction is exposed.
Every evaluator returns a scalar score and a small, actionable conflict witness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import hypot, isfinite


def _array(answer, key, n=None):
    if not isinstance(answer, dict) or not isinstance(answer.get(key), list):
        raise ValueError(f"Expected JSON object with list field {key}")
    xs = answer[key]
    if n is not None and len(xs) != n:
        raise ValueError(f"{key} must have length {n}")
    return xs


def _ints(xs, lo=None, hi=None, distinct=False):
    if any(type(x) is not int or (lo is not None and x < lo) or
           (hi is not None and x > hi) for x in xs):
        raise ValueError("Expected in-range integers")
    if distinct and len(xs) != len(set(xs)):
        raise ValueError("Expected distinct entries")


def _bits(xs, n):
    if len(xs) != n or any(type(x) is not int or x not in (0, 1) for x in xs):
        raise ValueError(f"Expected {n} binary entries")


def _points(answer, key, n, grid):
    pts = _array(answer, key, n)
    if any(not isinstance(p, list) or len(p) != 2 for p in pts):
        raise ValueError("Expected integer coordinate pairs")
    for p in pts:
        _ints(p, 0, grid)
    return pts


def _det(a):
    # Fraction-free Bareiss elimination, exact integer determinant.
    n = len(a); b = [row[:] for row in a]; sign = 1; prev = 1
    for k in range(n - 1):
        if b[k][k] == 0:
            q = next((q for q in range(k + 1, n) if b[q][k] != 0), None)
            if q is None: return 0
            b[k], b[q] = b[q], b[k]; sign = -sign
        pivot = b[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                b[i][j] = (b[i][j] * pivot - b[i][k] * b[k][j]) // prev
        for i in range(k + 1, n): b[i][k] = 0
        prev = pivot
    return sign * b[-1][-1]


def _score(task, answer):
    f, p = task["family"], task["parameters"]
    n = p.get("n")
    if f in {"weighted_maxcut", "graph_coloring", "independent_set",
             "dominating_set", "triangle_free_subgraph"}:
        edges = p["edges"]
        if f == "triangle_free_subgraph":
            mask = _array(answer, "selected_edges", len(edges)); _bits(mask, len(edges))
            chosen = {tuple(sorted((u, v))) for b, (u, v, _) in zip(mask, edges) if b}
            triangles = [(u, v, w) for u, v, w in combinations(range(n), 3)
                         if {(u, v), (v, w), (u, w)} <= chosen]
            return (len(triangles) * (len(edges) + 1) - sum(mask),
                    {"triangle_conflicts": triangles[:4], "selected_edges": sum(mask)})
        xs = _array(answer, "labels", n)
        if f == "graph_coloring":
            _ints(xs, 0, p["colors"] - 1)
            bad = [(u, v) for u, v, _ in edges if xs[u] == xs[v]]
            return sum(w for u, v, w in edges if xs[u] == xs[v]), {"monochromatic_edges": bad[:6]}
        _bits(xs, n)
        if f == "weighted_maxcut":
            return (sum(w for u, v, w in edges if xs[u] != xs[v]),
                    {"uncut_edges": [(u, v) for u, v, _ in edges if xs[u] == xs[v]][:6]})
        if f == "independent_set":
            bad = [(u, v) for u, v, _ in edges if xs[u] and xs[v]]
            return (n + 1) * len(bad) - sum(xs), {"edge_conflicts": bad[:6]}
        missing = [u for u in range(n) if not xs[u] and not any(
            xs[v] for a, b, _ in edges for v in ([b] if a == u else [a] if b == u else []))]
        return sum(xs) + (n + 1) * len(missing), {"uncovered_vertices": missing[:6]}

    if f in {"sidon_fixed", "golomb_ruler", "ap_free", "difference_basis",
             "cyclic_difference_set"}:
        xs = _array(answer, "set", p["k"]); _ints(xs, 0, p["limit"], True)
        xs = sorted(xs)
        if f == "sidon_fixed":
            sums = Counter(xs[i] + xs[j] for i in range(len(xs)) for j in range(i, len(xs)))
            return sum(v - 1 for v in sums.values()), {"repeated_sums": [k for k, v in sums.items() if v > 1][:6]}
        if f == "golomb_ruler":
            diffs = Counter(b - a for a, b in combinations(xs, 2))
            duplicate = sum(v - 1 for v in diffs.values())
            return ((p["limit"] + 1) * duplicate + (xs[-1] - xs[0]),
                    {"duplicate_differences": [k for k, v in diffs.items() if v > 1][:6]})
        if f == "ap_free":
            values = set(xs)
            bad = [(a, b, 2 * b - a) for a, b in combinations(xs, 2) if 2 * b - a in values]
            return len(bad), {"three_term_progressions": bad[:6]}
        if f == "difference_basis":
            diffs = {b - a for a, b in combinations(xs, 2)}
            missing = [d for d in range(1, p["target_difference"] + 1) if d not in diffs]
            return len(missing), {"missing_differences": missing[:8]}
        v = p["modulus"]
        count = Counter((a - b) % v for a in xs for b in xs if a != b)
        expected = p["lambda_target"]
        loss = sum(abs(count[d] - expected) for d in range(1, v))
        return loss, {"worst_difference_counts": sorted(
            ((d, count[d]) for d in range(1, v)),
            key=lambda item: -abs(item[1] - expected))[:6]}

    if f == "schur_coloring":
        colors = _array(answer, "colors", n); _ints(colors, 0, p["colors"] - 1)
        bad = [(x, y, x + y) for x in range(1, n + 1) for y in range(x, n - x + 1)
               if colors[x - 1] == colors[y - 1] == colors[x + y - 1]]
        return len(bad), {"monochromatic_relations": bad[:6]}

    if f in {"binary_code", "deletion_code", "constant_weight_code"}:
        words = _array(answer, "words", p["count"])
        if any(not isinstance(w, list) for w in words): raise ValueError("Expected list of words")
        for w in words:
            _bits(w, n)
            if f == "constant_weight_code" and sum(w) != p["weight"]:
                raise ValueError("Each word must have the fixed Hamming weight")
        if len({tuple(w) for w in words}) != len(words): raise ValueError("Words must be distinct")
        if f == "binary_code":
            pairs = [(sum(a != b for a, b in zip(words[i], words[j])), i, j)
                     for i, j in combinations(range(len(words)), 2)]
            minimum = min(pairs)
            return minimum[0], {"closest_pair": list(minimum[1:])}
        if f == "deletion_code":
            # Repeated deletion positions of the *same* word do not violate a
            # deletion code. Only a signature shared by distinct codewords does.
            owners = defaultdict(set)
            for word_id, word in enumerate(words):
                for position in range(n):
                    owners[tuple(word[:position] + word[position + 1:])].add(word_id)
            conflicts = [(signature, sorted(ids)) for signature, ids in owners.items()
                         if len(ids) > 1]
            return (sum(len(ids) - 1 for _, ids in conflicts),
                    {"collision_signatures": [
                        {"signature": list(signature), "word_ids": ids}
                        for signature, ids in conflicts[:4]]})
        pairs = [(sum(a and b for a, b in zip(words[i], words[j])), i, j)
                 for i, j in combinations(range(len(words)), 2)]
        worst = max(pairs)
        return worst[0], {"largest_overlap_pair": list(worst[1:])}

    if f == "n_queens":
        perm = _array(answer, "columns", n); _ints(perm, 0, n - 1, True)
        bad = [(i, j) for i, j in combinations(range(n), 2)
               if abs(i - j) == abs(perm[i] - perm[j])]
        return len(bad), {"diagonal_conflicts": bad[:6]}

    if f == "low_autocorrelation":
        xs = _array(answer, "signs", n)
        if any(type(x) is not int or x not in (-1, 1) for x in xs):
            raise ValueError("Expected ±1 signs")
        c = [sum(xs[i] * xs[i + d] for i in range(n - d)) for d in range(1, n)]
        return sum(v * v for v in c), {"largest_sidelobes": sorted(
            enumerate(c, 1), key=lambda kv: -abs(kv[1]))[:5]}

    if f in {"square_packing", "heilbronn_points", "circle_packing"}:
        pts = _points(answer, "points", n, p["grid"])
        if len({tuple(q) for q in pts}) != n and f != "circle_packing":
            raise ValueError("Distinct points required")
        if f == "heilbronn_points":
            if any(x + y > p["grid"] for x, y in pts):
                raise ValueError("Point lies outside the right triangle")
            areas = [(abs((b[0] - a[0]) * (c[1] - a[1]) -
                          (b[1] - a[1]) * (c[0] - a[0])), i, j, k)
                     for (i, a), (j, b), (k, c) in combinations(enumerate(pts), 3)]
            worst = min(areas)
            return worst[0], {"minimum_area_triple": list(worst[1:])}
        dist = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2, i, j)
                for (i, a), (j, b) in combinations(enumerate(pts), 2)]
        closest = min(dist)
        if f == "square_packing":
            return closest[0], {"closest_pair": list(closest[1:])}
        diameter_sq = (2 * p["radius"]) ** 2
        overlaps = [(i, j, diameter_sq - d) for d, i, j in dist if d < diameter_sq]
        boundary = [i for i, (x, y) in enumerate(pts)
                    if min(x, y, p["grid"] - x, p["grid"] - y) < p["radius"]]
        return (sum(v for _, _, v in overlaps) + diameter_sq * len(boundary),
                {"overlap_pairs": overlaps[:5], "boundary_conflicts": boundary[:5]})

    if f == "metric_tsp":
        route = _array(answer, "tour", n); _ints(route, 0, n - 1, True)
        pts = p["points"]
        length = sum(hypot(pts[route[i]][0] - pts[route[(i + 1) % n]][0],
                           pts[route[i]][1] - pts[route[(i + 1) % n]][1])
                     for i in range(n))
        return round(length, 9), {"longest_edges": sorted(
            ((round(hypot(pts[route[i]][0] - pts[route[(i + 1) % n]][0],
                           pts[route[i]][1] - pts[route[(i + 1) % n]][1]), 6),
              route[i], route[(i + 1) % n]) for i in range(n)), reverse=True)[:5]}

    if f == "max_determinant":
        a = _array(answer, "matrix", n)
        if any(not isinstance(row, list) or len(row) != n or
               any(type(x) is not int or x not in (-1, 1) for x in row) for row in a):
            raise ValueError("Expected n×n ±1 matrix")
        return abs(_det(a)), {"determinant": _det(a)}

    if f == "precedence_scheduling":
        order = _array(answer, "priority", n); _ints(order, 0, n - 1, True)
        done, machine = {}, [0] * p["machines"]
        rank = {x: i for i, x in enumerate(order)}
        left = set(range(n))
        while left:
            available = [j for j in left if all(k in done for k in p["predecessors"][j])]
            if not available: raise ValueError("Cyclic precedence instance")
            j = min(available, key=lambda x: rank[x]); m = min(range(len(machine)), key=lambda x: machine[x])
            start = max(machine[m], max((done[x] for x in p["predecessors"][j]), default=0))
            done[j] = start + p["durations"][j]; machine[m] = done[j]; left.remove(j)
        return max(done.values()), {"last_tasks": sorted(done.items(), key=lambda kv: -kv[1])[:5]}

    if f == "set_cover":
        chosen = _array(answer, "chosen_sets"); _ints(chosen, 0, len(p["sets"]) - 1, True)
        covered = set().union(*(set(p["sets"][i]) for i in chosen)) if chosen else set()
        missing = sorted(set(range(p["universe"])) - covered)
        return len(chosen) + (len(p["sets"]) + 1) * len(missing), {"uncovered_elements": missing[:8]}

    if f == "discrepancy":
        signs = _array(answer, "signs", n)
        if any(type(x) is not int or x not in (-1, 1) for x in signs):
            raise ValueError("Expected ±1 signs")
        sums = [sum(signs[j] for j in row) for row in p["set_system"]]
        return max(abs(s) for s in sums), {"worst_set": max(range(len(sums)), key=lambda j: abs(sums[j])),
                                        "worst_sum": max(sums, key=abs)}

    if f == "polynomial_certificate":
        coeffs = _array(answer, "coefficients", p["basis_count"]); _ints(coeffs, -9, 9)
        residual = [p["target"][j] - sum(coeffs[i] * p["basis"][i][j]
                                             for i in range(len(coeffs)))
                    for j in range(len(p["target"]))]
        return (100 * sum(abs(x) for x in residual) + sum(x != 0 for x in coeffs),
                {"residual_coefficients": residual})

    raise ValueError(f"Unknown family: {f}")


def verify(task, answer):
    """Return exact objective and a small error witness; invalid JSON gets no score."""
    try:
        score, feedback = _score(task, answer)
        if type(score) not in (int, float) or not isfinite(score):
            raise ValueError("Non-finite scalar objective")
        return {"valid": True, "score": score,
                "direction": task["objective"]["direction"], "feedback": feedback}
    except (KeyError, TypeError, ValueError) as error:
        return {"valid": False, "score": None,
                "direction": task["objective"]["direction"],
                "feedback": {"error": str(error)[:250]}}
