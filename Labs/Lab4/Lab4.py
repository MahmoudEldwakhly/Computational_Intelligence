import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define g1, g2, g3
def g(x):
    x1, x2, x3 = x
    g1 = 3*x1 - np.cos(x2 * x3) - 0.5
    g2 = x1**2 - 81*(x2 + 0.1)**2 + np.sin(x3) + 1.06
    g3 = np.exp(-x1*x2) + 20*x3 + (10*np.pi - 3)/3
    return np.array([g1, g2, g3])

# Objective wrapper
def F(x):
    return g(x)

# Initial guess
x0 = np.array([0.1, 0.1, -0.1])

res = least_squares(F, x0, method='lm')

res.x, res.cost


x1 = np.linspace(-1, 2, 60)
x2 = np.linspace(-1, 1, 60)
X1, X2 = np.meshgrid(x1, x2)

x3_sol = res.x[2]
G1 = 3*X1 - np.cos(X2 * x3_sol) - 0.5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, G1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("g1(x1,x2,x3_sol)")
plt.title("3D Surface of g1 using solved x3")
plt.show()


import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# 1. Define the nonlinear equations g1, g2, g3
# ---------------------------------------------------------
def g(x):
    x1, x2, x3 = x
    g1 = 3*x1 - np.cos(x2*x3) - 0.5
    g2 = x1**2 - 81*(x2 + 0.1)**2 + np.sin(x3) + 1.06
    g3 = np.exp(-x1*x2) + 20*x3 + (10*np.pi - 3)/3
    return np.array([g1, g2, g3])

# ---------------------------------------------------------
# 2. Objective function (least squares minimization)
# ---------------------------------------------------------
def F(x):
    return g(x)

# ---------------------------------------------------------
# 3. Initial guess
# ---------------------------------------------------------
x0 = np.array([0.1, 0.1, -0.1])

# ---------------------------------------------------------
# 4. Solve using SciPy Levenberg-Marquardt method
# ---------------------------------------------------------
res = least_squares(F, x0, method='lm')

print("Solution x =", res.x)
print("Residuals g(x) =", res.fun)
print("Cost (1/2 * sum g^2) =", res.cost)
print("Solver Message:", res.message)

# ---------------------------------------------------------
# 5. 3D Surface Plot of g1(x1, x2) at solved x3
# ---------------------------------------------------------
x1_vals = np.linspace(-1, 2, 80)
x2_vals = np.linspace(-1, 1, 80)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

x3_sol = res.x[2]
G1 = 3*X1 - np.cos(X2 * x3_sol) - 0.5

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, G1, cmap='viridis', alpha=0.8)
ax.scatter(res.x[0], res.x[1], 0, color="r", s=80, label="Solution")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("g1")
plt.title("3D surface of g1(x1, x2) at solved x3")
plt.legend()
plt.show()

# ---------------------------------------------------------
# 6. 2D Contour Plot of F(x1, x2) at solved x3
# ---------------------------------------------------------
F_map = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i,j], X2[i,j], x3_sol])
        F_map[i,j] = 0.5 * np.sum(g(x)**2)

plt.figure(figsize=(8,6))
cont = plt.contourf(X1, X2, F_map, levels=40, cmap='jet')
plt.colorbar(cont)
plt.scatter(res.x[0], res.x[1], color='white', s=90, label="Solution")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Contour plot of objective function F(x1, x2)")
plt.legend()
plt.show()
