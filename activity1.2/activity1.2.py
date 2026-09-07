import math


def fixed_point_iteration(g, x0, tol=1e-9, max_iter=2000, decimals=8, verbose=False):
    history = [x0]
    x = x0
    for n in range(1, max_iter + 1):
        x_new = g(x)
        history.append(x_new)
        if verbose:
            print(f"  step {n:3d}: x = {x_new:.{decimals}f}")
        if abs(x_new - x) < tol:
            return x_new, n, history
        x = x_new
    raise RuntimeError("Fixed-point iteration did not converge within max_iter steps")


def report(title, g, x0, decimals=8, tol=None):
    if tol is None:
        tol = 0.5 * 10 ** (-decimals - 1)
    root, steps, _ = fixed_point_iteration(g, x0, tol=tol, decimals=decimals)
    print(f"{title}")
    print(f"  Initial guess : x0 = {x0}")
    print(f"  Fixed point   : x  = {root:.{decimals}f}")
    print(f"  Steps needed  : {steps}")
    print()
    return root, steps


# ---------------------------------------------------------------------------
# Problem 1: Apply FPI to find the solution of each equation to 8 decimals.
# ---------------------------------------------------------------------------
def problem1():
    print("=" * 70)
    print("PROBLEM 1: Fixed-Point Iteration (8 correct decimal places)")
    print("=" * 70)

    g_a = lambda x: (2 * x + 2) ** (1 / 3)
    report("(a) x^3 = 2x + 2   =>  g(x) = (2x + 2)^(1/3)", g_a, x0=1.5)

    g_b = lambda x: math.log(7 - x)
    report("(b) e^x + x = 7    =>  g(x) = ln(7 - x)", g_b, x0=1.5)

    g_c = lambda x: math.log(4 - math.sin(x))
    report("(c) e^x + sin x = 4 =>  g(x) = ln(4 - sin x)", g_c, x0=1.0)


# ---------------------------------------------------------------------------
# Problem 2: Apply FPI to find the solution of each equation to 8 decimals.
# ---------------------------------------------------------------------------
def problem2():
    print("=" * 70)
    print("PROBLEM 2: Fixed-Point Iteration (8 correct decimal places)")
    print("=" * 70)

    g_a = lambda x: (1 - x) ** (1 / 5)
    report("(a) x^5 + x = 1    =>  g(x) = (1 - x)^(1/5)", g_a, x0=0.7)

    g_b = lambda x: (math.sin(x) - 5) / 6
    report("(b) sin x = 6x + 5 =>  g(x) = (sin x - 5) / 6", g_b, x0=-1.0)

    g_c = lambda x: math.sqrt(3 - math.log(x))
    report("(c) ln x + x^2 = 3 =>  g(x) = sqrt(3 - ln x)", g_c, x0=1.5)


# ---------------------------------------------------------------------------
# Problem 3: Square roots via Fixed-Point Iteration (Example 1.6 style)
#   g(x) = (x + a/x) / 2
# ---------------------------------------------------------------------------
def problem3():
    print("=" * 70)
    print("PROBLEM 3: Square roots by Fixed-Point Iteration")
    print("            g(x) = (x + a/x) / 2")
    print("=" * 70)

    for label, a, x0 in [("(a)", 3, 1.0), ("(b)", 5, 1.0)]:
        g = lambda x, a=a: (x + a / x) / 2
        report(f"{label} sqrt({a})", g, x0=x0)


# ---------------------------------------------------------------------------
# Problem 4: Cube roots via Fixed-Point Iteration
#   g(x) = (2x + A/x^2) / 3
# ---------------------------------------------------------------------------
def problem4():
    print("=" * 70)
    print("PROBLEM 4: Cube roots by Fixed-Point Iteration")
    print("            g(x) = (2x + A/x^2) / 3")
    print("=" * 70)

    for label, A, x0 in [("(a)", 2, 1.0), ("(b)", 3, 1.0), ("(c)", 5, 1.0)]:
        g = lambda x, A=A: (2 * x + A / x ** 2) / 3
        report(f"{label} cube root of {A}", g, x0=x0)


# ---------------------------------------------------------------------------
# Problem 5: Is g(x) = cos^2(x) a convergent FPI (as g(x) = cos x is)?
#   Find the fixed point to 6 correct decimals, report steps, and discuss
#   local convergence using |g'(x*)| < 1 (Theorem 1.6).
# ---------------------------------------------------------------------------
def problem5():
    print("=" * 70)
    print("PROBLEM 5: g(x) = cos^2(x) as a Fixed-Point Iteration")
    print("=" * 70)

    g = lambda x: math.cos(x) ** 2
    root, steps = report(
        "g(x) = cos^2(x)", g, x0=1.0, decimals=6, tol=0.5e-7
    )

    # Local convergence check: g'(x) = -2 cos(x) sin(x) = -sin(2x)
    gprime = lambda x: -math.sin(2 * x)
    slope = gprime(root)

    print("Local convergence analysis (Theorem 1.6):")
    print(f"  g'(x)      = -sin(2x)")
    print(f"  g'(x*)     = {slope:.6f}   at x* = {root:.6f}")
    print(f"  |g'(x*)|   = {abs(slope):.6f}")
    if abs(slope) < 1:
        print("  => |g'(x*)| < 1, so g(x) = cos^2(x) IS a locally convergent")
        print("     fixed-point iteration near x*, just like g(x) = cos x.")
    else:
        print("  => |g'(x*)| >= 1, so g(x) = cos^2(x) is NOT locally")
        print("     convergent near x*.")
    print()


if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
