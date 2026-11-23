"""Independent numeric verifiers for audited target-near training variants.

The functions below use only Python's standard library. They accept parsed
JSON objects. Invalid answers return a concrete conflict when possible.
This module contains no benchmark answer or incumbent construction.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import math
from pathlib import Path

_VENDOR_NAMESPACES = {}


def _vendor(name, dimension=None):
    """Load frozen local verifier mathematics; isolate per-parameter globals."""
    key=(name,dimension)
    if key not in _VENDOR_NAMESPACES:
        source=Path(__file__).with_name(name).read_text(encoding='utf-8')
        namespace={'__name__':'_trusted_training_verifier'}
        exec(compile(source,name,'exec'),namespace)
        if dimension is not None:namespace['N']=dimension
        _VENDOR_NAMESPACES[key]=namespace
    return _VENDOR_NAMESPACES[key]


def _done(task, value, feedback=None):
    return {"valid": True, "score": value,
            "direction": task["objective"]["direction"],
            "feedback": feedback or {"metric": task["objective"]["metric"]}}


def _bad(task, reason, **detail):
    return {"valid": False, "score": None,
            "direction": task["objective"]["direction"],
            "feedback": {"reason": reason, **detail}}


def _integer(x):
    return type(x) is int


def _real(x):
    return type(x) in (int, float) and math.isfinite(x)


def _bareiss(matrix):
    a = [row[:] for row in matrix]
    n = len(a)
    sign, prev = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            pivot = next((i for i in range(k + 1, n) if a[i][k]), None)
            if pivot is None:
                return 0
            a[k], a[pivot] = a[pivot], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * pivot - a[i][k] * a[k][j]
                if numerator % prev:
                    raise ArithmeticError("Nonexact Bareiss division")
                a[i][j] = numerator // prev
        for i in range(k + 1, n):
            a[i][k] = 0
        prev = pivot
    return sign * a[-1][-1]


def verify(task: dict, answer: dict) -> dict:
    """Return {valid,score,direction,feedback}; never execute model code."""
    if not isinstance(answer, dict):
        return _bad(task, "answer_must_be_object")
    f, p = task["family"], task["parameters"]
    try:
        if f == "covering_array":
            rows = answer["rows"]
            R, C, q, t, target = (p[k] for k in ("rows", "columns", "alphabet", "strength", "target_coverage"))
            if not isinstance(rows, list) or len(rows) != R or any(
                not isinstance(row, list) or len(row) != C or any(
                    not _integer(x) or x < 0 or x >= q for x in row) for row in rows):
                return _bad(task, "matrix_shape_or_symbol")
            loss, worst = 0, None
            for cols in combinations(range(C), t):
                counts = Counter(tuple(row[j] for j in cols) for row in rows)
                for tup in product(range(q), repeat=t):
                    defect = max(0, target - counts[tup])
                    loss += defect * defect
                    if defect and worst is None:
                        worst = {"columns": cols, "tuple": tup, "count": counts[tup]}
            return _done(task, loss, {"first_deficient_interaction": worst})
        if f == "labs":
            seq, n = answer["sequence"], p["length"]
            if not isinstance(seq, list) or len(seq) != n or any(type(x) is not int or x not in (-1, 1) for x in seq):
                return _bad(task, "binary_sequence_shape")
            corr = [sum(seq[i] * seq[i + k] for i in range(n - k)) for k in range(1, n)]
            return _done(task, sum(x*x for x in corr), {"largest_correlation_lag": 1 + max(range(n-1), key=lambda j: abs(corr[j]))})
        if f == "maxdet":
            a, n = answer["matrix"], p["order"]
            if not isinstance(a, list) or len(a) != n or any(
                not isinstance(row, list) or len(row) != n or any(type(x) is not int or x not in (-1,1) for x in row) for row in a):
                return _bad(task, "sign_matrix_shape")
            det = abs(_bareiss(a))
            return _done(task, det, {"exact_abs_determinant": det})
        if f == "ramsey":
            a, n, size = answer["adjacency_matrix"], p["vertices"], p["clique_order"]
            if not isinstance(a, list) or len(a) != n or any(not isinstance(row, list) or len(row) != n or any(type(x) is not int or x not in (0,1) for x in row) for row in a):
                return _bad(task, "adjacency_shape")
            if any(a[i][i] or any(a[i][j] != a[j][i] for j in range(i+1,n)) for i in range(n)):
                return _bad(task, "adjacency_not_symmetric_or_diagonal")
            all_edges = size * (size - 1) // 2
            loss, witness = 0, None
            for verts in combinations(range(n), size):
                e = sum(a[i][j] for i,j in combinations(verts, 2))
                if e in (0, all_edges):
                    loss += 1
                    if witness is None: witness = {"vertices": verts, "edge_count": e}
            return _done(task, loss, {"first_monochromatic_subset": witness})
        if f == "schur":
            colors, n, q = answer["colors_digits"], p["length"], p["colors"]
            if not isinstance(colors, str) or len(colors) != n or any(ch not in "0123456789"[:q] for ch in colors):
                return _bad(task, "color_string_shape")
            loss, witness = 0, None
            for x in range(1, n+1):
                for y in range(x, n-x+1):
                    if colors[x-1] == colors[y-1] == colors[x+y-1]:
                        loss += 1
                        if witness is None: witness = (x,y,x+y)
            return _done(task, loss, {"first_monochromatic_sum": witness})
        if f == "mols":
            squares, n, k = answer["squares"], p["order"], p["count"]
            if not isinstance(squares, list) or len(squares) != k or any(
                not isinstance(sq, list) or len(sq) != n or any(
                    not isinstance(row, list) or len(row) != n or set(row) != set(range(n)) or any(not _integer(x) for x in row)
                    for row in sq) for sq in squares):
                return _bad(task, "latin_square_shape_or_row")
            for si, sq in enumerate(squares):
                for c in range(n):
                    if {sq[r][c] for r in range(n)} != set(range(n)):
                        return _bad(task, "latin_column", square=si, column=c)
            defects = {}
            for a,b in combinations(range(k), 2):
                pairs = {(squares[a][r][c],squares[b][r][c]) for r in range(n) for c in range(n)}
                defects[f"{a},{b}"] = n*n - len(pairs)
            return _done(task, sum(defects.values()), {"pair_defects": defects})
        if f == "no_three_in_line":
            pts, n = answer["points"], p["grid_side"]
            if not isinstance(pts, list) or not pts or any(not isinstance(x, list) or len(x)!=2 or any(not _integer(y) or y<0 or y>=n for y in x) for x in pts):
                return _bad(task, "grid_points_shape")
            if len({tuple(x) for x in pts}) != len(pts):return _bad(task,"duplicate_point")
            if len(pts)>2*n:return _bad(task,"elementary_row_bound",maximum=2*n)
            for i,j,k in combinations(range(len(pts)),3):
                a,b,c=pts[i],pts[j],pts[k]
                if (b[0]-a[0])*(c[1]-a[1])==(b[1]-a[1])*(c[0]-a[0]):
                    return _bad(task,"collinear_triple",indices=[i,j,k])
            return _done(task,len(pts))
        if f == "heilbronn":
            pts,n=answer["points"],p["count"]
            if not isinstance(pts,list) or len(pts)!=n or any(not isinstance(z,list) or len(z)!=2 or any(not _real(a) for a in z) or z[0]<0 or z[1]<0 or z[0]+z[1]>1 for z in pts):
                return _bad(task,"points_outside_unit_triangle_or_shape")
            area2, witness = None, None
            for i,j,k in combinations(range(n),3):
                a,b,c=pts[i],pts[j],pts[k]
                v=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
                if area2 is None or v<area2: area2,witness=v,[i,j,k]
            # Normalized to triangle area: determinant is the area ratio.
            return _done(task,area2,{"minimum_area_triple":witness})
        if f == "kissing":
            vec,d,n=answer["vectors"],p["dimension"],p["count"]
            if not isinstance(vec,list) or len(vec)!=n or any(not isinstance(v,list) or len(v)!=d or any(not _real(x) for x in v) for v in vec):
                return _bad(task,"vector_shape")
            norms=[math.sqrt(sum(x*x for x in v)) for v in vec]
            if any(not math.isfinite(z) or z<=1e-12 for z in norms):return _bad(task,"zero_or_nonfinite_vector")
            unit=[[x/z for x in v] for v,z in zip(vec,norms)]
            loss,worst=0.0,None
            for i,j in combinations(range(n),2):
                dot=sum(a*b for a,b in zip(unit[i],unit[j]))
                distance=math.sqrt(max(0.0,8.0-8.0*dot))
                overlap=max(0.0,2.0-distance)
                loss+=overlap
                if overlap and (worst is None or overlap>worst[2]):worst=(i,j,overlap)
            return _done(task,loss,{"largest_overlap_pair":worst})
        if f == "shannon":
            words,q,d,max_words=answer["words"],p["alphabet"],p["length"],p["max_words"]
            if not isinstance(words,list) or not 1<=len(words)<=max_words or any(not isinstance(w,list) or len(w)!=d or any(not _integer(x) or x<0 or x>=q for x in w) for w in words):
                return _bad(task,"word_shape")
            if len({tuple(w) for w in words})!=len(words):return _bad(task,"duplicate_word")
            for i,j in combinations(range(len(words)),2):
                if all(min(abs(a-b),q-abs(a-b))<2 for a,b in zip(words[i],words[j])):
                    return _bad(task,"confusable_word_pair",indices=[i,j])
            return _done(task,len(words))
        if f == "difference_bases":
            vals,kmax,maxval=answer["set"],p["max_size"],p["max_value"]
            if not isinstance(vals,list) or not vals or len(vals)>kmax or any(not _integer(x) or x<0 or x>maxval for x in vals):
                return _bad(task,"basis_shape_or_bound")
            b=sorted(set(vals)|{0})
            if len(b)>kmax:return _bad(task,"basis_size_bound")
            diffs={x-y for x,y in combinations(b,2) if x>y} | {y-x for x,y in combinations(b,2) if y>x}
            v=0
            while v+1 in diffs:v+=1
            if not v:return _bad(task,"no_positive_difference_covered")
            return _done(task,len(b)*len(b)/v,{"contiguous_difference_coverage":v,"basis_size":len(b)})
        if f in {"tammes","thomson"}:
            vec,n=answer["vectors"],p["count"]
            if not isinstance(vec,list) or len(vec)!=n or any(not isinstance(v,list) or len(v)!=3 or any(not _real(x) for x in v) for v in vec):
                return _bad(task,"vector_shape")
            norms=[math.sqrt(sum(x*x for x in v)) for v in vec]
            if any(z<=1e-12 or not math.isfinite(z) for z in norms):return _bad(task,"zero_vector")
            unit=[[x/z for x in v] for v,z in zip(vec,norms)]
            distances=[math.dist(unit[i],unit[j]) for i,j in combinations(range(n),2)]
            if min(distances)<=1e-10:return _bad(task,"coincident_points")
            if f=="tammes":return _done(task,min(distances),{"minimum_distance":min(distances)})
            return _done(task,sum(1/x for x in distances),{"minimum_distance":min(distances)})
        if f == "sidon":
            vals,limit=answer["values"],p["max_value"]
            if not isinstance(vals,list) or not vals or any(not _integer(x) or x<1 or x>limit for x in vals) or vals!=sorted(set(vals)):
                return _bad(task,"sidon_values_shape")
            seen={}
            for i in range(len(vals)):
                for j in range(i,len(vals)):
                    s=vals[i]+vals[j]
                    if s in seen:return _bad(task,"repeated_pair_sum",first=seen[s],second=[i,j],sum=s)
                    seen[s]=[i,j]
            return _done(task,len(vals))
        if f == "van_der_waerden":
            digits,q,k,limit=answer["colors_digits"],p["colors"],p["progression_length"],p["max_length"]
            if not isinstance(digits,str) or not 1<=len(digits)<=limit or any(c not in "0123456789"[:q] for c in digits) or digits[0]!="0":
                return _bad(task,"color_string_shape_or_normalization")
            n=len(digits)
            for step in range(1,(n-1)//(k-1)+1):
                for start in range(n-(k-1)*step):
                    vals=[digits[start+j*step] for j in range(k)]
                    if len(set(vals))==1:return _bad(task,"monochromatic_progression",start=start+1,step=step,length=k)
            return _done(task,n)
        if f == "sorting_network":
            pairs,n,max_comps=answer["comparators"],p["wires"],p["max_comparators"]
            if not isinstance(pairs,list) or len(pairs)>max_comps or any(not isinstance(z,list) or len(z)!=2 or any(not _integer(x) for x in z) or not 0<=z[0]<z[1]<n for z in pairs):
                return _bad(task,"comparator_shape_or_bound")
            for bits in product((0,1),repeat=n):
                state=list(bits)
                for i,j in pairs:
                    if state[i]>state[j]:state[i],state[j]=state[j],state[i]
                if state!=sorted(state):return _bad(task,"binary_counterexample",input=list(bits),output=state)
            return _done(task,len(pairs))
        if f == "two_deletion_code":
            words,n,max_words=answer["words"],p["length"],p["max_words"]
            if not isinstance(words,list) or not 1<=len(words)<=max_words or any(not isinstance(w,str) or len(w)!=n or set(w)-set('01') for w in words):
                return _bad(task,"binary_word_shape")
            if len(set(words))!=len(words):return _bad(task,"duplicate_word")
            seen={}
            for wi,w in enumerate(words):
                for i,j in combinations(range(n),2):
                    child=w[:i]+w[i+1:j]+w[j+1:]
                    if child in seen and seen[child]!=wi:return _bad(task,"shared_two_deletion_descendant",word_indices=[seen[child],wi],descendant=child)
                    seen[child]=wi
            return _done(task,len(words))
        if f == "snake_in_box":
            actions,d=answer["actions"],p["dimension"]
            if not isinstance(actions,list) or not actions or len(actions)>(1<<d)-1 or any(not _integer(a) or not 0<=a<d for a in actions):
                return _bad(task,"action_shape")
            verts=[0]
            for a in actions:verts.append(verts[-1]^(1<<a))
            if len(set(verts))!=len(verts):return _bad(task,"revisited_vertex")
            for i in range(len(verts)):
                for j in range(i+2,len(verts)):
                    if (verts[i]^verts[j]).bit_count()==1:return _bad(task,"induced_path_chord",vertices=[i,j])
            return _done(task,len(actions))
        if f == "min_distance_ratio":
            pts,n=answer["vectors"],p["count"]
            if not isinstance(pts,list) or len(pts)!=n or any(not isinstance(v,list) or len(v)!=2 or any(not _real(x) for x in v) for v in pts):
                return _bad(task,"point_shape")
            ds=[(pts[i][0]-pts[j][0])**2+(pts[i][1]-pts[j][1])**2 for i,j in combinations(range(n),2)]
            if min(ds)<=1e-24:return _bad(task,"coincident_points")
            return _done(task,max(ds)/min(ds),{"minimum_distance_squared":min(ds),"maximum_distance_squared":max(ds)})
        if f in {"circle_packing","circles_rectangle"}:
            circles,n=answer["circles"],p["count"]
            if not isinstance(circles,list) or len(circles)!=n or any(not isinstance(z,list) or len(z)!=3 or any(not _real(x) for x in z) or z[2]<=0 for z in circles):
                return _bad(task,"circle_shape_or_nonpositive_radius")
            if f=="circle_packing":
                side=p["side"]
                for i,(x,y,r) in enumerate(circles):
                    if x-r < -1e-12 or y-r < -1e-12 or x+r>side+1e-12 or y+r>side+1e-12:
                        return _bad(task,"circle_outside_square",index=i)
            else:
                xmin=min(x-r for x,y,r in circles);xmax=max(x+r for x,y,r in circles)
                ymin=min(y-r for x,y,r in circles);ymax=max(y+r for x,y,r in circles)
                if (xmax-xmin)+(ymax-ymin)>p["width_plus_height"]+1e-12:
                    return _bad(task,"rectangle_perimeter_budget",width=xmax-xmin,height=ymax-ymin)
            nearest=None
            for i,j in combinations(range(n),2):
                a,b=circles[i],circles[j]
                gap=math.hypot(a[0]-b[0],a[1]-b[1])-a[2]-b[2]
                if nearest is None or gap<nearest[2]:nearest=(i,j,gap)
                if gap < -1e-12:return _bad(task,"overlapping_circles",indices=[i,j],overlap=-gap)
            return _done(task,sum(z[2] for z in circles),{"smallest_boundary_gap":nearest})
        if f == "erdos_min_overlap":
            vals,n=answer["values"],p["samples"]
            if not isinstance(vals,list) or len(vals)!=n or any(not _real(x) or not 0<=x<=1 for x in vals):
                return _bad(task,"step_function_shape_or_range")
            total=sum(vals)
            if total<=0:return _bad(task,"zero_integral")
            # The target rescales to integral 1 and rejects, rather than clips,
            # any value that leaves [0,1] after normalization.
            vals=[x*n/(2*total) for x in vals]
            if any(x>1+1e-12 for x in vals):return _bad(task,"normalization_exceeds_one")
            other=[1-x for x in vals]
            scores=[sum(vals[i]*other[i+k] for i in range(max(0,-k),min(n,n-k))) *2/n
                    for k in range(-n+1,n)]
            return _done(task,max(scores),{"worst_shift":scores.index(max(scores))-(n-1)})
        if f in {"first_autocorr","third_autocorr"}:
            vals,n=answer["values"],p["samples"]
            if not isinstance(vals,list) or len(vals)!=n or any(not _real(x) or abs(x)>1e6 for x in vals):
                return _bad(task,"autocorrelation_values_shape")
            if f=="first_autocorr" and any(x<0 for x in vals):return _bad(task,"negative_value")
            integral=sum(vals)
            if abs(integral)<1e-10:return _bad(task,"zero_integral")
            conv=[0.0]*(2*n-1)
            for i,a in enumerate(vals):
                for j,b in enumerate(vals):conv[i+j]+=a*b
            # Target discretization uses dx=0.5/n. The numerator has one dx,
            # whereas the squared integral has dx^2, leaving a 1/dx factor.
            value=(2*n)*max(conv)/(integral*integral)
            if f=="third_autocorr":value=abs(value)
            return _done(task,value,{"peak_lag":conv.index(max(conv))-(n-1)})
        if f == "flat_polynomial":
            coeff,n=answer["coefficients"],p["degree"]+1
            if not isinstance(coeff,list) or len(coeff)!=n or any(type(x) is not int or x not in (-1,1) for x in coeff):
                return _bad(task,"sign_coefficients_shape")
            samples=p["circle_samples"]
            peak=0.0
            for j in range(samples):
                z=complex(math.cos(2*math.pi*j/samples),math.sin(2*math.pi*j/samples))
                value=0j
                for a in coeff:value=value*z+a
                peak=max(peak,abs(value))
            return _done(task,peak/math.sqrt(n+1),{"peak_sampled_modulus":peak,"sample_count":samples})
        if f == "spencer_discrepancy":
            a,n=answer["matrix"],p["order"]
            if not isinstance(a,list) or len(a)!=n or any(not isinstance(row,list) or len(row)!=n or any(type(x) is not int or x not in (-1,1) for x in row) for row in a):
                return _bad(task,"sign_matrix_shape")
            disc=min(max(abs(sum(a[i][j]*x[j] for j in range(n))) for i in range(n))
                     for x in product((-1,1),repeat=n))
            return _done(task,disc/math.sqrt(n),{"exact_discrepancy":disc})
        if f == "ring_loading":
            pairs,n=answer["pairs"],p["pair_count"]
            if not isinstance(pairs,list) or len(pairs)!=n or any(not isinstance(z,list) or len(z)!=2 or any(not _real(x) or x<0 for x in z) or sum(z)>1+1e-12 for z in pairs):
                return _bad(task,"pair_shape_or_budget")
            worst_choice=None;best=float('inf')
            for bits in product((0,1),repeat=n):
                z=[pairs[i][1] if bits[i] else -pairs[i][0] for i in range(n)]
                total=sum(z);prefix=0.0;peak=0.0
                for x in z:
                    prefix+=x;peak=max(peak,abs(2*prefix-total))
                if peak<best:best,worst_choice=peak,bits
            return _done(task,best,{"minimizing_adversary_choice":worst_choice})
        if f == "second_autocorr":
            vals,n=answer["values"],p["samples"]
            if not isinstance(vals,list) or len(vals)!=n or any(not _real(x) or x < -1e-6 or x>1e6 for x in vals):
                return _bad(task,"nonnegative_values_shape")
            vals=[max(0.0,x) for x in vals]
            if sum(vals)<=0:return _bad(task,"zero_integral")
            conv=[0.0]*(2*n-1)
            for i,a in enumerate(vals):
                for j,b in enumerate(vals):conv[i+j]+=a*b
            y=[0.0,*conv,0.0]
            interval=1/(2*n)
            l2_sq=sum(interval*(y[i]*y[i]+y[i]*y[i+1]+y[i+1]*y[i+1])/3
                      for i in range(2*n))
            norm1=sum(abs(v) for v in conv)/(2*n)
            norm_inf=max(abs(v) for v in conv)
            return _done(task,l2_sq/(norm1*norm_inf),{"peak_convolution":norm_inf})
        if f == "sidon_45":
            vals,n,limit=answer["set"],p["count"],p["max_abs_value"]
            if not isinstance(vals,list) or len(vals)!=n or any(not _integer(x) or abs(x)>limit for x in vals) or len(set(vals))!=n:
                return _bad(task,"integer_set_shape_or_bound")
            for quad in combinations(range(n),4):
                differences={abs(vals[i]-vals[j]) for i,j in combinations(quad,2)}
                if len(differences)<5:return _bad(task,"not_4_5_set",indices=quad,distinct_differences=len(differences))
            # Exact search is bounded to n<=15 for these training variants.
            largest=0
            for size in range(n,0,-1):
                found=False
                for subset in combinations(vals,size):
                    sums=[a+b for i,a in enumerate(subset) for b in subset[i:]]
                    if len(sums)==len(set(sums)):
                        found=True;break
                if found:largest=size;break
            return _done(task,largest/n,{"largest_sidon_subset_size":largest})
        if f == "kakeya":
            offsets,n=answer["offsets"],p["triangle_count"]
            if not isinstance(offsets,list) or len(offsets)!=n or any(not isinstance(x,str) or len(x)>32 for x in offsets):
                return _bad(task,"decimal_offset_shape")
            namespace=_vendor('vendor_kakeya.py',n)
            score=float(namespace['evaluate']({'offsets':offsets}))
            if not math.isfinite(score):return _bad(task,"nonfinite_area")
            return _done(task,score,{"exact_rational_union_area_evaluated":True})
        if f == "edges_vs_triangles":
            rows,bins,max_rows=answer["weights"],p["bins"],p["max_rows"]
            if not isinstance(rows,list) or not 1<=len(rows)<=max_rows or any(
                not isinstance(row,list) or len(row)!=bins or any(not _real(x) or x<0 for x in row) or sum(row)<=1e-7 for row in rows):
                return _bad(task,"probability_row_shape_or_nonnegative")
            import numpy as np
            namespace=_vendor('vendor_edges.py')
            matrix=np.array([[x/sum(row) for x in row] for row in rows],dtype=np.float64)
            rho,tri=namespace['sum_pairwise_triple_products_batch'](matrix)
            area,gap=namespace['analyze_density_curve'](rho,tri)
            score=-(float(area)+10*float(gap))
            if not math.isfinite(score):return _bad(task,"nonfinite_density_curve")
            return _done(task,score,{"area":float(area),"maximum_gap":float(gap)})
        if f == "pnt_certificate":
            raw,max_key,max_entries,range_multiplier=answer["partial_function"],p["max_key"],p["max_entries"],p["range_multiplier"]
            if not isinstance(raw,dict) or not raw or len(raw)>max_entries:
                return _bad(task,"partial_function_size")
            coeff={}
            for key,value in raw.items():
                if not isinstance(key,str) or not key.isdecimal() or not _real(value):
                    return _bad(task,"noninteger_key_or_nonfinite_value")
                k=int(key)
                if not 1<=k<=max_key:return _bad(task,"key_out_of_range",key=key)
                coeff[k]=min(10.0,max(-10.0,float(value)))
            total=sum(value/k for k,value in coeff.items())
            coeff[1]=coeff.get(1,0.0)-total
            # Unlike the target's 10^7 random samples, exact finite checking
            # tests every integer breakpoint where floor(x/k) may change.
            upper=range_multiplier*max_key
            worst=(0,0.0)
            for x in range(1,upper+1):
                value=sum(v*(x//k) for k,v in coeff.items())
                if value>worst[1]:worst=(x,value)
                if value>1.0001:return _bad(task,"certificate_constraint",x=x,value=value)
            score=-sum(value*math.log(k)/k for k,value in coeff.items())
            return _done(task,score,{"worst_checked_x":worst[0],"maximum_constraint_value":worst[1],"exact_finite_check_end":upper})
        return _bad(task,"unknown_family",family=f)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError,
            ZeroDivisionError, AssertionError, ArithmeticError) as exc:
        return _bad(task,"malformed_or_unscorable",error_type=type(exc).__name__)
