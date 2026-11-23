# Source: EinsteinArena frozen verifier kakeya-needle-128; no target answer included.
import math
from fractions import Fraction

N = 128
MAX_FRAC_DIGITS = 25
MAX_INT_DIGITS = 4


def _parse_offset(s):
    if not isinstance(s, str):
        raise ValueError("Each offset must be a base-10 decimal string.")
    t = s.strip()
    neg = t.startswith("-")
    if neg or t.startswith("+"):
        t = t[1:]
    if t.count(".") > 1:
        raise ValueError("Malformed decimal string: " + repr(s))
    intpart, _, fracpart = t.partition(".")
    digits = intpart + fracpart
    if digits == "" or not digits.isdigit():
        raise ValueError("Malformed decimal string: " + repr(s))
    if len(fracpart) > MAX_FRAC_DIGITS:
        raise ValueError("Offset has more than " + str(MAX_FRAC_DIGITS) + " fractional digits: " + repr(s))
    if len(intpart) > MAX_INT_DIGITS:
        raise ValueError("Offset integer part is too large: " + repr(s))
    val = Fraction(int(digits), 10 ** len(fracpart))
    return -val if neg else val


def _triangle_endpoint_lines(scaled_xs, den):
    # The two section endpoints of T_j are affine in y. Each is stored as
    # (slope, intercept), both scaled by N*den so they are exact integers.
    lines = []
    for index, x in enumerate(scaled_xs, start=1):
        intercept = N * x
        lines.append((den * index, intercept))
        lines.append((den * (index - 1), intercept + den))
    return lines


def _collect_breakpoints(lines):
    # Every height in (0,1) where two endpoint lines cross, as an exact
    # rational p/q in lowest terms. No epsilon: nothing is merged or dropped.
    pts = set()
    m = len(lines)
    for a in range(m):
        slope_a, intercept_a = lines[a]
        for b in range(a + 1, m):
            slope_delta = slope_a - lines[b][0]
            if slope_delta == 0:
                continue
            p = lines[b][1] - intercept_a
            q = slope_delta
            if q < 0:
                p = -p
                q = -q
            if 0 < p < q:
                g = math.gcd(p, q)
                pts.add((p // g, q // g))
    out = sorted(pts, key=lambda t: Fraction(t[0], t[1]))
    out.append((1, 1))
    return out


def _union_interval_length(intervals):
    intervals.sort()
    total = 0
    cur_l, cur_r = intervals[0]
    for left, right in intervals[1:]:
        if left > cur_r:
            total += cur_r - cur_l
            cur_l, cur_r = left, right
        elif right > cur_r:
            cur_r = right
    total += cur_r - cur_l
    return total


def _union_length_at_y(lines, p, q):
    # Union length at y = p/q, returned as an integer numerator over q*N*den.
    intervals = []
    for k in range(0, len(lines), 2):
        slope_l, intercept_l = lines[k]
        slope_r, intercept_r = lines[k + 1]
        intervals.append((intercept_l * q + slope_l * p,
                          intercept_r * q + slope_r * p))
    return _union_interval_length(intervals)


def triangle_union_area(xs):
    # Put every offset over one denominator so all coordinates are integers.
    den = 1
    for x in xs:
        den = den // math.gcd(den, x.denominator) * x.denominator
    scaled_xs = [int(x * den) for x in xs]

    lines = _triangle_endpoint_lines(scaled_xs, den)
    scale = N * den

    total = Fraction(0)
    prev_y = Fraction(0)
    prev_u = Fraction(_union_length_at_y(lines, 0, 1), scale)
    for p, q in _collect_breakpoints(lines):
        y = Fraction(p, q)
        u = Fraction(_union_length_at_y(lines, p, q), q * scale)
        # Exact: the union length is affine across each slab.
        total += (prev_u + u) * (y - prev_y) / 2
        prev_y = y
        prev_u = u
    return total


def evaluate(data):
    offsets = data["offsets"]
    if len(offsets) != N:
        raise ValueError("Expected exactly " + str(N) + " offsets, got " + str(len(offsets)))

    xs = [_parse_offset(s) for s in offsets]
    # A common horizontal translation does not change the union area.
    x0 = xs[0]
    xs = [x - x0 for x in xs]

    area = triangle_union_area(xs)

    # The union cannot exceed the total area of the 128 triangles.
    if area <= 0 or area > Fraction(1, 2):
        raise ValueError("Internal area check failed: " + str(float(area)))
    return float(area)
