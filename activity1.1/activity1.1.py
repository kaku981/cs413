import math
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def bisection(f, a, b, decimals=6, max_iter=200, verbose=False):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, 0, [(0, a, b, a, fa)]
    if fb == 0:
        return b, 0, [(0, a, b, b, fb)]
    if fa * fb > 0:
        raise ValueError(f"f(a) and f(b) must have opposite signs. "
                          f"f({a})={fa:.6g}, f({b})={fb:.6g}")

    tol = 0.5 * 10 ** (-decimals)
    table = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        table.append((i, a, b, c, fc))
        if verbose:
            print(f"iter {i:3d}: a={a:.10f} b={b:.10f} c={c:.10f} f(c)={fc:.3e}")

        if fc == 0 or (b - a) / 2 < tol:
            return c, i, table

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return (a + b) / 2, max_iter, table


def print_result(name, f, a, b, decimals):
    root, iters, _ = bisection(f, a, b, decimals=decimals)
    print(f"{name}: interval=({a}, {b})  root ≈ {round(root, decimals)}  "
          f"(f(root)={f(root):.2e}, iterations={iters})")
    return root

def find_sign_change_intervals(f, x_min, x_max, n=2000):
    xs = np.linspace(x_min, x_max, n)
    intervals = []
    prev_x, prev_f = xs[0], f(xs[0])
    for x in xs[1:]:
        fx = f(x)
        if np.isfinite(prev_f) and np.isfinite(fx) and prev_f * fx < 0:
            intervals.append((prev_x, x))
        prev_x, prev_f = x, fx
    return intervals


def plot_function(f, x_min, x_max, title, filename, roots=None, n=1000):
    xs = np.linspace(x_min, x_max, n)
    ys = [f(x) for x in xs]
    plt.figure(figsize=(7, 5))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.plot(xs, ys, label='f(x)')
    if roots:
        for r in roots:
            plt.plot(r, f(r), 'ro')
            plt.annotate(f'{r:.4f}', (r, f(r)), textcoords="offset points",
                         xytext=(5, 8), fontsize=9)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  -> saved plot: {filename}")


# ----------------------------------------------------------------------
# PROBLEM 1: root to 6 correct decimal places
# ----------------------------------------------------------------------
def problem1():
    print("\n=== Problem 1 (6 decimal places) ===")

    f1a = lambda x: x**3 - 9
    print_result("1(a) x^3 = 9", f1a, 2, 3, 6)

    f1b = lambda x: 3*x**3 + x**2 - x - 5
    print_result("1(b) 3x^3+x^2=x+5", f1b, 1, 2, 6)

    f1c = lambda x: math.cos(x)**2 - x + 6
    print_result("1(c) cos^2(x)+6=x", f1c, 6, 7, 6)


# ----------------------------------------------------------------------
# PROBLEM 2: root to 8 correct decimal places
# ----------------------------------------------------------------------
def problem2():
    print("\n=== Problem 2 (8 decimal places) ===")

    f2a = lambda x: x**5 + x - 1
    print_result("2(a) x^5+x=1", f2a, 0, 1, 8)

    f2b = lambda x: math.sin(x) - 6*x - 5
    print_result("2(b) sin(x)=6x+5", f2b, -1, 0, 8)

    f2c = lambda x: math.log(x) + x**2 - 3
    print_result("2(c) ln(x)+x^2=3", f2c, 1, 2, 8)


# ----------------------------------------------------------------------
# PROBLEM 3: locate ALL solutions, plot, pick 3 unit-length intervals,
# then solve to 6 decimal places
# ----------------------------------------------------------------------
def problem3():
    print("\n=== Problem 3 (locate all roots, plot, solve to 6 dp) ===")

    equations = {
        "3(a) 2x^3 - 6x - 1 = 0": (lambda x: 2*x**3 - 6*x - 1, -3, 3),
        "3(b) e^(x-2) + x^3 - x = 0": (lambda x: math.exp(x - 2) + x**3 - x, -3, 3),
        "3(c) 1 + 5x - 6x^3 - e^(2x) = 0": (lambda x: 1 + 5*x - 6*x**3 - math.exp(2*x), -3, 3),
    }

    for name, (f, xmin, xmax) in equations.items():
        print(f"\n{name}")
        intervals = find_sign_change_intervals(f, xmin, xmax)
        print(f"  Sign-change intervals found: "
              f"{[(round(a,2), round(b,2)) for a,b in intervals]}")

        roots = []
        for (a, b) in intervals:
            a_round, b_round = math.floor(a), math.ceil(b)
            try:
                root, iters, _ = bisection(f, a, b, decimals=6)
            except ValueError:
                continue
            roots.append(root)
            print(f"  Root in ({a:.4f}, {b:.4f}) [unit interval "
                  f"({a_round}, {b_round})] -> x ≈ {round(root, 6)} "
                  f"(iterations={iters})")

        filename = os.path.join(
            SCRIPT_DIR, f"{name.split()[0].replace('(', '').replace(')', '')}.png"
        )
        plot_function(f, xmin, xmax, name, filename, roots=roots)

if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
