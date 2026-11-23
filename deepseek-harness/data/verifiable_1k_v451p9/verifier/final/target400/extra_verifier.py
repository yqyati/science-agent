"""Exact, self-contained verifiers for seven L2-adjacent training families.

Every problem changes the benchmark instance. No target answer or incumbent is
read. Returned score is a genuine numeric objective for a valid construction.
"""
from __future__ import annotations

from itertools import combinations
import heapq
import math
from graph_iso import connected, isomorphic, deck_overlap
from uncertainty_lp import score as uncertainty_score
from fractions import Fraction


def _permutation(values, items):
    return isinstance(values, list) and len(values) == len(items) and \
        all(type(v) is int for v in values) and set(values) == set(items)


def _invalid(reason, direction="minimize"):
    return {"valid": False, "score": None, "direction": direction,
            "feedback": {"reason": reason}}


def _ok(score, detail=None, direction="minimize"):
    return {"valid": True, "score": int(score), "direction": direction,
            "feedback": detail or {}}


def tsp(spec, answer):
    p = spec["parameters"]
    n, xy = p["n"], p["coords"]
    tour = answer.get("tour") if isinstance(answer, dict) else None
    if not _permutation(tour, range(n)):
        return _invalid("tour must be a permutation of all city labels")
    length = sum(abs(xy[a][0]-xy[b][0])+abs(xy[a][1]-xy[b][1])
                 for a,b in zip(tour, tour[1:]+tour[:1]))
    return _ok(length, {"closed_tour_length": length})


def _tree_edges(n, seq):
    deg = [1]*n
    for x in seq: deg[x] += 1
    leaves = [i for i,d in enumerate(deg) if d == 1]
    heapq.heapify(leaves)
    edges = []
    for x in seq:
        leaf = heapq.heappop(leaves)
        edges.append((leaf,x))
        deg[leaf] -= 1; deg[x] -= 1
        if deg[x] == 1: heapq.heappush(leaves,x)
    a,b = heapq.heappop(leaves),heapq.heappop(leaves)
    edges.append((a,b))
    return edges


def graceful(spec, answer):
    p = spec["parameters"];n=p["n"]
    labels = answer.get("labels") if isinstance(answer,dict) else None
    if not _permutation(labels, range(n)):
        return _invalid("labels must be a permutation of 0..n-1")
    diffs = {abs(labels[a]-labels[b]) for a,b in _tree_edges(n,p["prufer"])}
    defect = n-1-len(diffs)
    return _ok(defect, {"distinct_edge_differences":len(diffs),
                        "missing_difference_count":defect})


def cvrp(spec, answer):
    p=spec["parameters"];n=p["n"]
    tour=answer.get("tour") if isinstance(answer,dict) else None
    if not _permutation(tour, range(1,n+1)):
        return _invalid("tour must permute customer labels 1..n")
    xy=p["coords"];demand=p["demands"];cap=p["capacity"]
    depot=p["depot"]
    def dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
    load=0;last=depot;length=0;routes=0
    for customer in tour:
        need=demand[customer-1]
        if need>cap:return _invalid("one customer exceeds capacity")
        if load+need>cap:
            length+=dist(last,depot);load=0;last=depot
        if load==0:routes+=1
        point=xy[customer-1]
        length+=dist(last,point);last=point;load+=need
    length+=dist(last,depot)
    return _ok(length,{"route_count":routes,"travel_length":length})


def rcpsp(spec, answer):
    p=spec["parameters"];n=p["n"]
    priority=answer.get("priority") if isinstance(answer,dict) else None
    if not _permutation(priority,range(1,n-1)):
        return _invalid("priority must permute all real jobs")
    preds=p["predecessors"];dur=p["durations"];req=p["demands"];cap=p["capacities"]
    done={0:0};starts={};unscheduled=set(priority)
    horizon=sum(dur)+1
    usage=[[0]*horizon for _ in cap]
    while unscheduled:
        ready=next((j for j in priority if j in unscheduled and
                    all(x in done for x in preds[j])),None)
        if ready is None:return _invalid("cyclic or unreachable precedence")
        earliest=max((done[x] for x in preds[ready]),default=0)
        for t in range(earliest,horizon-dur[ready]+1):
            if all(all(usage[h][u]+req[ready][h]<=cap[h]
                       for u in range(t,t+dur[ready])) for h in range(len(cap))):
                break
        else:return _invalid("schedule horizon exceeded")
        starts[ready]=t;done[ready]=t+dur[ready];unscheduled.remove(ready)
        for h in range(len(cap)):
            for u in range(t,t+dur[ready]):usage[h][u]+=req[ready][h]
    makespan=max(done.values());total=sum(starts.values())
    return _ok(100000*makespan+total,{"makespan":makespan,"start_sum":total})


def projective(spec, answer):
    p=spec["parameters"];n=p["n"];k=p["k"]
    matchings=answer.get("matchings") if isinstance(answer,dict) else None
    if not isinstance(matchings,list) or len(matchings)!=k:
        return _invalid("exactly k matchings required")
    images=[]
    for obj in matchings:
        if not isinstance(obj,dict):return _invalid("matching is not an object")
        if obj.get("kind")=="affine_mod_n":
            a,b=obj.get("a"),obj.get("b")
            if type(a) is not int or type(b) is not int or not 0<=b<n or math.gcd(a,n)!=1:
                return _invalid("invalid affine matching")
            one=[(a*i+b)%n for i in range(n)]
        elif obj.get("kind")=="explicit":
            one=obj.get("images")
        else:return _invalid("unknown matching kind")
        if not _permutation(one,range(n)):
            return _invalid("matching images must be a permutation")
        images.append(one)
    rows=[set(one[i] for one in images) for i in range(n)]
    if any(len(row)!=k for row in rows):
        return _invalid("matchings collide at a line")
    cycle=0;worst=[]
    for i in range(n):
        for j in range(i+1,n):
            overlap=len(rows[i]&rows[j]);cycle+=overlap*(overlap-1)//2
            if overlap>1 and len(worst)<4:worst.append([i,j,overlap])
    return _ok(cycle,{"four_cycle_count":cycle,"first_conflicts":worst})


def golomb(spec,answer):
    n=spec["parameters"]["n"]
    marks=answer.get("marks") if isinstance(answer,dict) else None
    if not isinstance(marks,list) or len(marks)!=n or any(type(x) is not int for x in marks) or \
            marks[0]!=0 or any(a>=b for a,b in zip(marks,marks[1:])):
        return _invalid("marks must be strictly increasing n integers starting at 0")
    diffs=[b-a for a,b in combinations(marks,2)]
    if len(set(diffs))!=len(diffs):
        return _invalid("repeated pairwise difference")
    return _ok(marks[-1],{"ruler_length":marks[-1]})


def union_closed(spec,answer):
    p=spec["parameters"];n=p["n"];m=p["m"]
    values=answer.get("family") if isinstance(answer,dict) else None
    if not isinstance(values,list) or len(values)!=m or any(type(x) is not int or x<0 or x>=1<<n for x in values):
        return _invalid("family must contain exactly m valid integer masks")
    if len(set(values))!=m or 0 not in values or \
            not all(any(v>>i&1 for v in values) for i in range(n)):
        return _invalid("masks must be distinct, contain empty set and use all elements")
    members=set(values)
    for a in values:
        for b in values:
            if a|b not in members:return _invalid("union closure violated")
    for i in range(n):
        for j in range(i+1,n):
            if not any(bool(v>>i&1)!=bool(v>>j&1) for v in values):
                return _invalid("family does not separate elements")
    freq=[sum(bool(v>>i&1) for v in values) for i in range(n)]
    return _ok(max(freq),{"max_frequency":max(freq),"frequencies":freq})


def sidorenko_k23(spec,answer):
    p=spec["parameters"];n=p["n"];m=p["m"]
    edges=answer.get("edges") if isinstance(answer,dict) else None
    if not isinstance(edges,list) or len(edges)!=m:
        return _invalid("exactly m edges required")
    pairs=[]
    for pair in edges:
        if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair) or \
                not 0<=pair[0]<pair[1]<n:
            return _invalid("edges must be distinct normalized pairs")
        pairs.append(tuple(pair))
    if len(set(pairs))!=m:return _invalid("duplicate edges")
    neighbors=[0]*n
    for u,v in pairs:neighbors[u]|=1<<v;neighbors[v]|=1<<u
    H=sum((neighbors[u]&neighbors[v]).bit_count()**3
          for u in range(n) for v in range(n))
    a,b,c,d,e,mod=p["weight_coefficients"]
    W=sum(1+(a*(u+1)**2+b*(v+1)**2+c*(u+1)*(v+1)+d*u+e*v)%mod
          for u,v in pairs)
    score=p["lexicographic_coefficient"]*H-W
    return _ok(score,{"common_neighbor_cube_sum":H,"edge_tie_weight":W})


def conway_shortage(spec,answer):
    p=spec["parameters"];n=p["n"];d=p["d"]
    rows=answer.get("adjacency_rows_hex") if isinstance(answer,dict) else None
    width=(n+3)//4
    if not isinstance(rows,list) or len(rows)!=n or any(
        not isinstance(x,str) or len(x)!=width or
        any(ch not in "0123456789abcdef" for ch in x) for x in rows):
        return _invalid("wrong adjacency_rows_hex format")
    bits=[int(x,16) for x in rows]
    if any(x>>n or x>>i&1 or x.bit_count()!=d for i,x in enumerate(bits)):
        return _invalid("out of range, self loop or wrong degree")
    if any(bool(bits[i]>>j&1)!=bool(bits[j]>>i&1)
           for i in range(n) for j in range(i+1,n)):
        return _invalid("adjacency matrix not symmetric")
    shortage=0;conflicts=[]
    for i in range(n):
        for j in range(i+1,n):
            shared=(bits[i]&bits[j]).bit_count()
            threshold=p["adjacent_threshold"] if bits[i]>>j&1 else p["nonadjacent_threshold"]
            loss=max(0,threshold-shared)
            shortage+=loss
            if loss and len(conflicts)<4:conflicts.append([i,j,shared,threshold])
    return _ok(shortage,{"shortage":shortage,"first_conflicts":conflicts})


def perfect_cuboid(spec,answer):
    p=spec["parameters"]
    if not isinstance(answer,dict):return _invalid("expected {a,b,c}")
    a,b,c=(answer.get(k) for k in ("a","b","c"))
    if any(type(x) is not int for x in (a,b,c)) or not p["lower"]<=a<b<c<p["upper"]:
        return _invalid("primitive ordered integer edges outside training interval")
    if math.gcd(math.gcd(a,b),c)!=1:return _invalid("edges must be primitive")
    face=[a*a+b*b,a*a+c*c,b*b+c*c]
    for value in face:
        s=math.isqrt(value)
        if s*s!=value:return _invalid("a face diagonal is not integral")
    Q=a*a+b*b+c*c;s=math.isqrt(Q)
    defect=min(Q-s*s,(s+1)*(s+1)-Q)
    return _ok(defect,{"nearest_square_defect":defect,"body_square":Q})


def sidorenko_mobius(spec,answer):
    p=spec["parameters"];n=p["n"];m=p["m"]
    rows=answer.get("adjacency_rows_hex") if isinstance(answer,dict) else None
    width=(n+3)//4
    if not isinstance(rows,list) or len(rows)!=n or any(
            not isinstance(x,str) or len(x)!=width or
            any(ch not in '0123456789abcdef' for ch in x) for x in rows):
        return _invalid("wrong adjacency_rows_hex format")
    bits=[int(x,16) for x in rows]
    if any(x>>n or x>>i&1 for i,x in enumerate(bits)):
        return _invalid("out of range or self loop")
    if any(bool(bits[i]>>j&1)!=bool(bits[j]>>i&1)
           for i in range(n) for j in range(i+1,n)):
        return _invalid("adjacency matrix not symmetric")
    if sum(x.bit_count() for x in bits)!=2*m:return _invalid("wrong edge count")
    # H has L_i--R_j whenever j-i mod 5 is 0,1,2. Fix five host images for
    # the L vertices. For each R_j, the number of legal images is the triple
    # intersection of neighborhoods of L_j,L_(j-1),L_(j-2).
    hom=0
    for a in range(n):
      for b in range(n):
       for c in range(n):
        for d in range(n):
         for e in range(n):
          L=(a,b,c,d,e)
          ways=1
          for j in range(5):
              ways*=(bits[L[j]]&bits[L[(j-1)%5]]&bits[L[(j-2)%5]]).bit_count()
              if not ways:break
          hom+=ways
    return _ok(hom,{"exact_hom_count":hom})


def graph_reconstruction(spec,answer):
    p=spec["parameters"];n=p["n"];m=p["m"]
    if not isinstance(answer,dict):return _invalid("expected G and H", "maximize")
    def parse(value):
        if not isinstance(value,dict) or value.get("n")!=n:return None
        edges=value.get("edges")
        if not isinstance(edges,list) or len(edges)!=m:return None
        pairs=[]
        for edge in edges:
            if not isinstance(edge,list) or len(edge)!=2 or any(type(x) is not int for x in edge) or \
                    not 0<=edge[0]<edge[1]<n:return None
            pairs.append(tuple(edge))
        if pairs!=sorted(set(pairs)):return None
        rows=[0]*n
        for u,v in pairs:rows[u]|=1<<v;rows[v]|=1<<u
        return rows
    g=parse(answer.get("G"));h=parse(answer.get("H"))
    if g is None or h is None:return _invalid("each graph needs n and exactly m sorted edges", "maximize")
    if sorted(x.bit_count() for x in g)!=sorted(x.bit_count() for x in h):
        return _invalid("degree multisets differ", "maximize")
    if not all(connected(rows) and connected(rows,True) for rows in (g,h)):
        return _invalid("graph or complement disconnected", "maximize")
    try:
        if isomorphic(g,h):return _invalid("graphs are isomorphic", "maximize")
        overlap,compatible=deck_overlap(g,h)
    except TimeoutError:
        return {"valid":None,"score":None,"direction":"maximize",
                "feedback":{"reason":"exact graph-isomorphism search budget exceeded"}}
    return _ok(overlap,{"matched_cards":overlap,"compatible_card_pairs":compatible},
               direction="maximize")


def uncertainty_lp(spec,answer):
    p=spec["parameters"]
    values=answer.get("laguerre_double_roots") if isinstance(answer,dict) else None
    if not isinstance(values,list) or len(values)!=1 or type(values[0]) not in (int,float):
        return _invalid("exactly one numeric double root is required")
    z=values[0]
    if not math.isfinite(z) or not p["lower"]<=z<=p["upper"]:
        return _invalid("double root outside the training interval")
    try:
        value=uncertainty_score(Fraction(str(z)))
    except (ArithmeticError,ValueError,ZeroDivisionError,OverflowError) as error:
        return _invalid(str(error)[:180])
    return {"valid":True,"score":value,"direction":"minimize",
            "feedback":{"upper_bound":value,
                        "certification":"exact rational Laguerre polynomial coefficients; float root approximation isolated between cubic derivative critical points"}}


VERIFIERS={"metric_tsp":tsp,"graceful_tree":graceful,"cvrp":cvrp,
           "rcpsp":rcpsp,"projective_plane":projective,
           "golomb_ruler":golomb,"union_closed":union_closed,
           "sidorenko_k23":sidorenko_k23,"conway_shortage":conway_shortage,
           "perfect_cuboid":perfect_cuboid,"sidorenko_mobius":sidorenko_mobius}
VERIFIERS["graph_reconstruction"]=graph_reconstruction
VERIFIERS["uncertainty_lp"]=uncertainty_lp


def verify(task,answer):
    family=task.get("family")
    if family not in VERIFIERS:return _invalid("unknown verifier family")
    return VERIFIERS[family](task,answer)
