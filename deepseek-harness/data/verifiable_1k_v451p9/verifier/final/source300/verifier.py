"""Standalone exact / deterministic numeric-feedback verifiers for source300.

The code deliberately does not import the evaluation benchmarks or historical
answers. A structurally valid but mathematically imperfect candidate receives
a scalar defect score and a concrete conflicting tuple for local repair.
"""
from __future__ import annotations

from itertools import combinations
import math


def _int(x):
    return type(x) is int


def _bad(message):
    return {"valid": False, "score": None, "direction": None,
            "feedback": {"error": message}}


def _ok(score, direction, witness=None, conflicts=0):
    assert math.isfinite(float(score))
    return {"valid": True, "score": score, "direction": direction,
            "feedback": {"conflict_count": conflicts, "witness": witness}}


def _conflict_score(p, conflicts, size, witness):
    """Three local repair signals, all zero exactly when no conflicts remain."""
    kind = p.get("loss_kind", "count")
    if kind == "count":
        score = len(conflicts)
    else:
        load = [0] * size
        for group in conflicts:
            for i in set(group):
                load[i] += 1
        if kind == "squared_load":
            score = sum(x*x for x in load)
        elif kind == "worst_load":
            score = max(load, default=0)
        else:
            return _bad("Unsupported loss_kind")
    return _ok(score, "minimize", witness, len(conflicts))


def _unwrap(answer, key="answer"):
    if isinstance(answer, dict):
        return answer.get(key, answer.get("answer"))
    return answer


def _mols(p, ans):
    n, k = p["n"], p["k"]
    sq = _unwrap(ans, "squares")
    if not isinstance(sq, list) or len(sq) != k:
        return _bad(f"Expected exactly {k} Latin squares")
    for h, L in enumerate(sq):
        if not isinstance(L, list) or len(L) != n or any(not isinstance(r, list) or
            len(r) != n or any(not _int(v) or not 0 <= v < n for v in r) for r in L):
            return _bad(f"Square {h} must be an {n}x{n} integer array")
        target = set(range(n))
        for i in range(n):
            if set(L[i]) != target or {L[j][i] for j in range(n)} != target:
                return _bad(f"Square {h} is not Latin at row/column {i}")
    if len({tuple(tuple(row) for row in L) for L in sq}) != k:
        return _bad("Latin squares must be distinct")
    defect, witness, pair_defects = 0, None, []
    for a, b in combinations(range(k), 2):
        count = {}
        for i in range(n):
            for j in range(n):
                pair = (sq[a][i][j], sq[b][i][j])
                count[pair] = count.get(pair, 0) + 1
        local=n*n-len(count)
        defect += local
        pair_defects.append(local)
        if witness is None:
            witness = next(({"squares": [a, b], "symbol_pair": list(key),
                              "multiplicity": value} for key, value in count.items()
                            if value > 1), None)
    mode=p.get("loss_kind","count")
    score=(defect if mode=="count" else sum(x*x for x in pair_defects)
           if mode=="squared_load" else max(pair_defects,default=0)
           if mode=="worst_load" else None)
    return _ok(score,"minimize",witness,defect) if score is not None else _bad("Unsupported loss_kind")


def _schur(p, ans):
    N, k = p["N"], p["k"]
    colors = _unwrap(ans, "colors")
    if not isinstance(colors, list) or len(colors) != N or any(not _int(c) or
        not 0 <= c < k for c in colors):
        return _bad(f"Expected {N} colors in 0..{k-1}")
    conflicts, witness = [], None
    for x in range(1, N//2 + 1):
        cx = colors[x-1]
        for y in range(x, N-x+1):
            z = x+y
            if cx == colors[y-1] == colors[z-1]:
                conflicts.append((x-1,y-1,z-1))
                if witness is None:
                    witness = {"x": x, "y": y, "z": z, "color": cx}
    return _conflict_score(p,conflicts,N,witness)


def _kissing(p, ans):
    d, m, c = p["d"], p["m"], p["cos100"]/100.0
    vectors = _unwrap(ans, "vectors")
    if not isinstance(vectors, list) or len(vectors) != m or any(
        not isinstance(v, list) or len(v) != d or
        any(not _int(x) or abs(x)>1000 for x in v) or
        all(x==0 for x in v) for v in vectors):
        return _bad(f"Expected {m} nonzero integer vectors of dimension {d}")
    norms = [math.sqrt(sum(x*x for x in v)) for v in vectors]
    target_distance = math.sqrt(8-8*c)
    score, conflicts, worst, gaps = 0.0, 0, None, []
    for i, j in combinations(range(m), 2):
        co = sum(a*b for a,b in zip(vectors[i],vectors[j]))/(norms[i]*norms[j])
        co = max(-1.0,min(1.0,co))
        gap = max(0.0,target_distance-math.sqrt(max(0.0,8-8*co)))
        if gap>1e-12:
            score += gap; conflicts += 1; gaps.append(gap)
            if worst is None or gap>worst[0]: worst=(gap,i,j,co)
    witness = None if worst is None else {"vector_indices":[worst[1],worst[2]],
                                         "cosine":worst[3],"distance_shortfall":worst[0]}
    mode=p.get("loss_kind","count")
    value=(sum(gaps) if mode=="count" else sum(x*x for x in gaps)
           if mode=="squared_load" else max(gaps,default=0.0)
           if mode=="worst_load" else None)
    return _ok(round(value,12),"minimize",witness,conflicts) if value is not None else _bad("Unsupported loss_kind")


def _orient(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _heilbronn(p, ans):
    n, T = p["n"], p["T"]
    pts = _unwrap(ans,"points")
    if not isinstance(pts,list) or len(pts)!=n or any(not isinstance(v,list) or
        len(v)!=2 or any(not _int(x) for x in v) for v in pts):
        return _bad(f"Expected {n} integer coordinate pairs")
    if len({tuple(v) for v in pts})!=n:
        return _bad("Points must be distinct")
    whole=_orient(*T)
    if whole==0: return _bad("Degenerate container triangle")
    for i,v in enumerate(pts):
        parts=[_orient(T[0],T[1],v),_orient(T[1],T[2],v),_orient(T[2],T[0],v)]
        if any(x*whole<0 for x in parts):
            return _bad(f"Point {i} lies outside the container triangle")
    areas=sorted((abs(_orient(pts[i],pts[j],pts[k])),[i,j,k])
                 for i,j,k in combinations(range(n),3))
    mode=p.get("loss_kind","count")
    take=1 if mode=="count" else 3 if mode=="squared_load" else 5 if mode=="worst_load" else 0
    if not take:return _bad("Unsupported loss_kind")
    value=sum(item[0] for item in areas[:take])/(take*abs(whole))
    return _ok(value,"maximize",{"point_indices":areas[0][1],
               "twice_area":areas[0][0]}, 1 if areas[0][0]==0 else 0)


def _words(p, ans, alphabet):
    q,d,m=p["q"],p["d"],p["m"]
    words=_unwrap(ans,"words")
    if not isinstance(words,list) or len(words)!=m or any(not isinstance(w,str)
        or len(w)!=d or any(ch not in alphabet for ch in w) for w in words):
        return None, _bad(f"Expected {m} words of length {d} over {alphabet}")
    if len(set(words))!=m:
        return None, _bad("Words must be distinct")
    return words,None


def _shannon(p, ans):
    q=p["q"]
    words,error=_words(p,ans,"012345678"[:q])
    if error:return error
    conflicts,witness=[],None
    for i,j in combinations(range(len(words)),2):
        if all(min(abs(int(x)-int(y)),q-abs(int(x)-int(y)))<=1
               for x,y in zip(words[i],words[j])):
            conflicts.append((i,j))
            if witness is None:witness={"adjacent_word_indices":[i,j]}
    return _conflict_score(p,conflicts,len(words),witness)


def _progression(p,ans):
    q,d,m=p["q"],p["d"],p["m"]
    words,error=_words(p,ans,"012345678"[:q])
    if error:return error
    S={tuple(map(int,w)):i for i,w in enumerate(words)}
    inv2=(q+1)//2
    conflicts,witness=[],None
    seen=set()
    for i,j in combinations(range(m),2):
        x,z=tuple(map(int,words[i])),tuple(map(int,words[j]))
        y=tuple(((a+b)*inv2)%q for a,b in zip(x,z))
        k=S.get(y)
        if k is not None and k not in (i,j):
            key=tuple(sorted((i,k,j)))
            if key not in seen:
                seen.add(key)
                conflicts.append(key)
                if witness is None:witness={"progression_indices":[i,k,j]}
    return _conflict_score(p,conflicts,m,witness)


def _corners(p,ans):
    n,m=p["n"],p["m"]
    pts=_unwrap(ans,"points")
    if not isinstance(pts,list) or len(pts)!=m or any(not isinstance(v,list) or
        len(v)!=2 or any(not _int(x) or not 0<=x<n for x in v) for v in pts):
        return _bad(f"Expected {m} distinct integer grid points in 0..{n-1}")
    S={tuple(v) for v in pts}
    if len(S)!=m:return _bad("Points must be distinct")
    conflicts,witness=[],None
    index={v:i for i,v in enumerate(tuple(v) for v in pts)}
    for x,y in S:
        for d in range(-n+1,n):
            if d and (x+d,y) in S and (x,y+d) in S:
                conflicts.append((index[(x,y)],index[(x+d,y)],index[(x,y+d)]))
                if witness is None:witness={"corner":[[x,y],[x+d,y],[x,y+d]]}
    return _conflict_score(p,conflicts,m,witness)


def _lineq(p,ans):
    n,m,a,b=p["n"],p["m"],p["a"],p["b"]
    values=_unwrap(ans,"set")
    if not isinstance(values,list) or len(values)!=m or any(not _int(x) or
        not 1<=x<=n for x in values):
        return _bad(f"Expected {m} integers from 1..{n}")
    S=set(values)
    if len(S)!=m:return _bad("Integers must be distinct")
    conflicts,witness=[],None
    index={v:i for i,v in enumerate(values)}
    for x in S:
        for y in S:
            u=a*x+b*y
            if u%(a+b)==0:
                z=u//(a+b)
                if z in S and not x==y==z:
                    conflicts.append((index[x],index[y],index[z]))
                    if witness is None:witness={"x":x,"y":y,"z":z}
    return _conflict_score(p,conflicts,m,witness)


def _trifference(p,ans):
    words,error=_words({"q":3,"d":p["d"],"m":p["m"]},ans,"012")
    if error:return error
    sep=p["separation"]
    conflicts,witness=[],None
    for i,j,k in combinations(range(len(words)),3):
        s=sum(len({a,b,c})==3 for a,b,c in zip(words[i],words[j],words[k]))
        if s<sep:
            conflicts.extend([(i,j,k)]*(sep-s))
            if witness is None:witness={"word_indices":[i,j,k],
                                        "separating_coordinates":s,"required":sep}
    return _conflict_score(p,conflicts,len(words),witness)


def _sidon_from_lineq(p,ans):
    n,m=p['n'],p['m']
    values=_unwrap(ans,'values')
    if not isinstance(values,list) or len(values)!=m or any(not _int(x) or
        not 1<=x<=n for x in values) or any(a>=b for a,b in zip(values,values[1:])):
        return _bad(f'Expected {m} strictly increasing integers in 1..{n}')
    sums={};conflicts=[];witness=None
    for i in range(m):
        for j in range(i,m):
            total=values[i]+values[j]
            for u,v in sums.get(total,[]):
                conflicts.append((u,v,i,j))
                if witness is None:
                    witness={'pair_indices':[[u,v],[i,j]],'repeated_sum':total}
            sums.setdefault(total,[]).append((i,j))
    return _conflict_score(p,conflicts,m,witness)


def _no3inline_from_corners(p,ans):
    n,m=p['n'],p['m']
    pts=_unwrap(ans,'points')
    if not isinstance(pts,list) or len(pts)!=m or any(not isinstance(v,list) or
        len(v)!=2 or any(not _int(x) or not 0<=x<n for x in v) for v in pts):
        return _bad(f'Expected {m} integer grid points in 0..{n-1}')
    if len({tuple(v) for v in pts})!=m:return _bad('Points must be distinct')
    conflicts=[];witness=None
    for i,j,k in combinations(range(m),3):
        if _orient(pts[i],pts[j],pts[k])==0:
            conflicts.append((i,j,k))
            if witness is None:witness={'collinear_point_indices':[i,j,k]}
    return _conflict_score(p,conflicts,m,witness)


def _deletion_from_trifference(p,ans):
    d,m=p['d'],p['m']
    words=_unwrap(ans,'words')
    if not isinstance(words,list) or len(words)!=m or any(not isinstance(w,str)
        or len(w)!=d or set(w)-{'0','1'} for w in words):
        return _bad(f'Expected {m} binary words of length {d}')
    if len(set(words))!=m:return _bad('Words must be distinct')
    descendants={};pairs=set();witness=None
    for i,w in enumerate(words):
        local={w[:a]+w[a+1:b]+w[b+1:] for a,b in combinations(range(d),2)}
        for sub in local:
            for j in descendants.get(sub,[]):
                pair=(j,i)
                pairs.add(pair)
                if witness is None:witness={'word_indices':[j,i],
                                            'common_two_deletion_descendant':sub}
            descendants.setdefault(sub,[]).append(i)
    return _conflict_score(p,sorted(pairs),m,witness)


def _covering_array_from_covering(p,ans):
    r,k,q,t,lam=(p[z] for z in ('rows','columns','q','strength','lambda'))
    rows=_unwrap(ans,'rows')
    if not isinstance(rows,list) or len(rows)!=r or any(not isinstance(row,list)
        or len(row)!=k or any(not _int(v) or not 0<=v<q for v in row) for row in rows):
        return _bad(f'Expected {r} rows of {k} integers in 0..{q-1}')
    from itertools import product
    per_block=[]; witness=None; missing=0
    for J in combinations(range(k),t):
        seen={}
        for row in rows:
            key=tuple(row[j] for j in J)
            seen[key]=seen.get(key,0)+1
        block=0
        for a in product(range(q),repeat=t):
            deficit=max(0,lam-seen.get(a,0))
            if deficit:
                missing+=1
                block+=deficit*deficit
                if witness is None:
                    witness={'columns':list(J),'tuple':list(a),
                             'observed':seen.get(a,0),'required':lam}
        per_block.append(block)
    mode=p.get('loss_kind','count')
    score=(sum(per_block) if mode=='count' else sum(x*x for x in per_block)
           if mode=='squared_load' else max(per_block,default=0)
           if mode=='worst_load' else None)
    return _ok(score,'minimize',witness,missing) if score is not None else _bad('Unsupported loss_kind')


def _det_bareiss(matrix):
    """Fraction-free exact determinant, including row-pivot swaps."""
    A=[row[:] for row in matrix];n=len(A);sign=1;prev=1
    for k in range(n-1):
        if A[k][k]==0:
            pivot=next((i for i in range(k+1,n) if A[i][k]!=0),None)
            if pivot is None:return 0
            A[k],A[pivot]=A[pivot],A[k];sign=-sign
        pivot=A[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                num=A[i][j]*pivot-A[i][k]*A[k][j]
                assert num%prev==0
                A[i][j]=num//prev
            A[i][k]=0
        prev=pivot
    return sign*A[-1][-1]


def _maxdet_from_matmul(p,ans):
    n=p['n'];matrix=_unwrap(ans,'matrix')
    if not isinstance(matrix,list) or len(matrix)!=n or any(not isinstance(row,list)
        or len(row)!=n or any(type(v) is not int or v not in (-1,1) for v in row)
        for row in matrix):
        return _bad(f'Expected a {n}x{n} sign matrix')
    value=abs(_det_bareiss(matrix))
    mode=p.get('loss_kind','count')
    score=(value if mode=='count' else value/(n**(n/2))
           if mode=='squared_load' else value*value if mode=='worst_load' else None)
    return _ok(score,'maximize',{'absolute_determinant':value,
        'singular':value==0},1 if value==0 else 0) if score is not None else _bad('Unsupported loss_kind')


def _vdw_from_apfree(p,ans):
    N,L=p['N'],p['length']
    digits=_unwrap(ans,'colors_digits')
    if not isinstance(digits,str) or len(digits)!=N or set(digits)-{'0','1'} or digits[0]!='0':
        return _bad(f'Expected {N} binary color digits starting with 0')
    conflicts=[];witness=None
    for d in range(1,(N-1)//(L-1)+1):
        for a in range(1,N-(L-1)*d+1):
            pos=tuple(a-1+j*d for j in range(L))
            if len({digits[i] for i in pos})==1:
                conflicts.append(pos)
                if witness is None:witness={'progression_start':a,'difference':d,
                                            'length':L,'color':digits[pos[0]]}
    return _conflict_score(p,conflicts,N,witness)


DISPATCH={"mols":_mols,"schur":_schur,"kissing":_kissing,
          "kissing_theta":_kissing,"heilbronn_shape":_heilbronn,
          "shannon":_shannon,"apfree_q":_progression,"capset":_progression,
          "corners":_corners,"lineq":_lineq,"trifference":_trifference,
          "sidon_from_lineq":_sidon_from_lineq,
          "no3inline_from_corners":_no3inline_from_corners,
          "deletion_from_trifference":_deletion_from_trifference}
DISPATCH.update({"covering_array_from_covering":_covering_array_from_covering,
                 "maxdet_from_matmul":_maxdet_from_matmul,
                 "vdw_from_apfree":_vdw_from_apfree,
                 "no3inline_from_capset":_no3inline_from_corners})


def verify(task, answer):
    """Return {valid, score, direction, feedback}; no external service needed."""
    family=task["family"]
    if family not in DISPATCH:
        return _bad(f"Unsupported source300 family {family}")
    try:
        return DISPATCH[family](task["parameters"],answer)
    except (TypeError,ValueError,KeyError,OverflowError) as exc:
        return _bad(f"Malformed candidate: {type(exc).__name__}: {exc}")
