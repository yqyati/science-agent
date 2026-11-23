"""One-double-root Laguerre LP training verifier with exact coefficients.

This is a small, reliable subfamily of the Einstein uncertainty-principle
construction. It computes the auxiliary polynomial in rational arithmetic and
approximates its largest positive sign-changing quotient root between the
critical points of the cubic. The floating root is not an exact certificate.
"""
from __future__ import annotations

from fractions import Fraction as Q
import math


def _choose(x,k):
    out=Q(1)
    for j in range(k):out*=x-j;out/=j+1
    return out


def laguerre(n):
    alpha=Q(-1,2)
    return [Q((-1)**j)*_choose(n+alpha,n-j)/math.factorial(j)
            for j in range(n+1)]


def _poly_at(poly,x):
    v=Q(0)
    for c in reversed(poly):v=v*x+c
    return v


def _solve(matrix,rhs):
    a=[list(row)+[b] for row,b in zip(matrix,rhs)]
    n=len(a)
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:raise ValueError('singular root constraint')
        a[j],a[pivot]=a[pivot],a[j]
        c=a[j][j];a[j]=[x/c for x in a[j]]
        for i in range(n):
            if i==j:continue
            c=a[i][j]
            if c:a[i]=[x-c*y for x,y in zip(a[i],a[j])]
    return [a[i][-1] for i in range(n)]


def quotient(z):
    polynomials=[laguerre(n) for n in (0,2,4,6)]
    matrix=[]
    for condition in ('at_zero','prime_zero','at_z','prime_z'):
        row=[]
        for p in polynomials:
            derivative=[j*p[j] for j in range(1,len(p))]
            row.append({'at_zero':lambda:p[0],
                        'prime_zero':lambda:derivative[0] if derivative else Q(0),
                        'at_z':lambda:_poly_at(p,z),
                        'prime_z':lambda:_poly_at(derivative,z)}[condition]())
        matrix.append(row)
    coeffs=_solve(matrix,[Q(0),Q(1),Q(0),Q(0)])
    g=[Q(0)]*7
    for c,p in zip(coeffs,polynomials):
        for j,value in enumerate(p):g[j]+=c*value
    # Divide by x(x-z)^2; exact zero remainder proves the prescribed roots.
    divisor=[Q(0),z*z,-2*z,Q(1)]
    rem=g[:];out=[Q(0)]*4
    for k in range(3,-1,-1):
        c=rem[k+3];out[k]=c
        for j in range(4):rem[k+j]-=c*divisor[j]
    if any(rem):raise ArithmeticError('Laguerre root constraints did not divide exactly')
    return out


def largest_positive_root(poly):
    # q is cubic. Between derivative critical points it is monotone, so every
    # sign change is isolated without missing a narrow interval on a grid.
    a,b,c,d=map(float,poly)
    if abs(d)<1e-20:raise ValueError('degenerate quotient')
    critical=[]
    disc=(2*c)**2-4*(3*d)*b
    if disc>=0:
        critical=[(-2*c-math.sqrt(disc))/(6*d),
                  (-2*c+math.sqrt(disc))/(6*d)]
    bound=1+max(abs(a/d),abs(b/d),abs(c/d))
    points=[0.0]+sorted(x for x in critical if 0<x<bound)+[bound]
    def val(x):return ((d*x+c)*x+b)*x+a
    roots=[]
    for lo,hi in zip(points,points[1:]):
        flo,fhi=val(lo),val(hi)
        if flo==0 and lo>0:roots.append(lo)
        if flo*fhi>=0:continue
        for _ in range(90):
            mid=(lo+hi)/2
            fm=val(mid)
            if (fm>0)==(flo>0):lo=mid;flo=fm
            else:hi=mid
        roots.append((lo+hi)/2)
    if not roots:raise ValueError('no positive sign-changing quotient root')
    return max(roots)


def score(z):
    return largest_positive_root(quotient(z))/(2*math.pi)
