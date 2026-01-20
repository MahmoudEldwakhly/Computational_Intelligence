import numpy as np

def solve_linear_system(m, n):
    """
    Solves the system A*x = b for given m and n
    where A and b are generated randomly.
    """

    # Random coefficients between -1 and 1
    A = np.random.uniform(-1, 1, (m, n))
    
    # Random constants between -1 and 3
    b = np.random.uniform(-1, 3, (m, 1))
    
    print(f"\nMatrix A ({m}x{n}):\n", A)
    print(f"\nVector b ({m}x1):\n", b)

    # Case 1: square system (m == n)
    if m == n:
        try:
            x = np.linalg.solve(A, b)
            print("\nSolution x:\n", x)
        except np.linalg.LinAlgError:
            print("\nMatrix A is singular — cannot solve exactly.")
    
    # Case 2: overdetermined system (m > n)
    else:
        # Use least squares method
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        print("\nLeast squares solution x:\n", x)
        print("Residuals:", residuals)

    print("-" * 60)
    return x


# Given cases
print("=== Case 1: m=3, n=3 ===")
solve_linear_system(3, 3)

print("\n=== Case 2: m=10, n=3 ===")
solve_linear_system(10, 3)


