"""Exact small-graph isomorphism and deck overlap for training variants.

Adapted from the project's reviewed OpenProblemBench graph-reconstruction
verifier, with an n<=12 training guard. No heuristic proxy is used for score.
"""
from __future__ import annotations

from collections import Counter


def connected(rows, complement=False):
    n=len(rows);full=(1<<n)-1;seen=1;frontier=1
    while frontier:
        bit=frontier&-frontier;frontier^=bit;u=bit.bit_length()-1
        neighbors=(full&~(rows[u]|1<<u)) if complement else rows[u]
        new=neighbors&~seen;seen|=new;frontier|=new
    return seen==full


def without(rows,deleted):
    keep=[v for v in range(len(rows)) if v!=deleted]
    return tuple(sum((1<<j) for j,v in enumerate(keep) if rows[u]>>v&1)
                 for u in keep)


def fingerprint(rows):
    d=[x.bit_count() for x in rows]
    return tuple(sorted((d[u],tuple(sorted(d[v] for v in range(len(rows))
                                            if rows[u]>>v&1))) for u in range(len(rows))))


def isomorphic(left,right,max_nodes=500000):
    if len(left)!=len(right) or fingerprint(left)!=fingerprint(right):return False
    n=len(left);cl=[x.bit_count() for x in left];cr=[x.bit_count() for x in right]
    for _ in range(n+1):
        sig=[]
        for rows,colors in ((left,cl),(right,cr)):
            sig.extend((colors[u],tuple(sorted(colors[v] for v in range(n)
                          if rows[u]>>v&1))) for u in range(n))
        palette={x:i for i,x in enumerate(sorted(set(sig)))}
        nl=[palette[cl[u],tuple(sorted(cl[v] for v in range(n)
                           if left[u]>>v&1))] for u in range(n)]
        nr=[palette[cr[u],tuple(sorted(cr[v] for v in range(n)
                           if right[u]>>v&1))] for u in range(n)]
        if Counter(nl)!=Counter(nr):return False
        if nl==cl and nr==cr:break
        cl,cr=nl,nr
    group={}
    for v,c in enumerate(cr):group.setdefault(c,[]).append(v)
    mapping=[-1]*n;used=[False]*n;nodes=0
    def dfs(depth):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise TimeoutError('exact graph isomorphism budget exceeded')
        if depth==n:return True
        best=None
        for u in range(n):
            if mapping[u]>=0:continue
            options=[]
            for v in group[cl[u]]:
                if used[v]:continue
                if all((left[u]>>x&1)==(right[v]>>y&1)
                       for x,y in enumerate(mapping) if y>=0):options.append(v)
            if not options:return False
            if best is None or len(options)<len(best[1]):best=(u,options)
        u,options=best
        for v in options:
            mapping[u]=v;used[v]=True
            if dfs(depth+1):return True
            mapping[u]=-1;used[v]=False
        return False
    return dfs(0)


def deck_overlap(g,h):
    n=len(g);left=[without(g,i) for i in range(n)]
    right=[without(h,j) for j in range(n)]
    fp_right={}
    for j,r in enumerate(right):fp_right.setdefault(fingerprint(r),[]).append(j)
    compat=[]
    cache={}
    for card in left:
        options=[]
        for j in fp_right.get(fingerprint(card),[]):
            key=(card,right[j])
            if key not in cache:cache[key]=isomorphic(*key)
            if cache[key]:options.append(j)
        compat.append(options)
    match=[-1]*n
    def augment(u,seen):
        for v in compat[u]:
            if v in seen:continue
            seen.add(v)
            if match[v]<0 or augment(match[v],seen):match[v]=u;return True
        return False
    return sum(augment(u,set()) for u in range(n)),sum(map(len,compat))
