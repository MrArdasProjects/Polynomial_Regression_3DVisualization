import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("Football_players.csv", encoding="ISO-8859-1")
x1 = df["Age"].to_numpy()
x2 = df["Skill"].to_numpy()
y = df["Salary"].to_numpy()


def build_polynomial_features(x1, x2, degree):
    features = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            features.append((x1 ** i) * (x2 ** j))
    return np.column_stack(features)


def compute_beta(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X2 = build_polynomial_features(x1, x2, degree=2)
beta2 = compute_beta(X2, y)
y_pred2 = X2 @ beta2
mse2 = mean_squared_error(y, y_pred2)


X3 = build_polynomial_features(x1, x2, degree=3)
beta3 = compute_beta(X3, y)
y_pred3 = X3 @ beta3
mse3 = mean_squared_error(y, y_pred3)


print("MSE with degree 2:", mse2)
print("MSE with degree 3:", mse3)


fig = plt.figure(figsize=(14, 6))


ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x1, x2, y, c='r', marker='o', label="Actual Salary")
ax1.plot_trisurf(x1, x2, y_pred2, alpha=0.5, color='gray')
ax1.set_xlabel("Age")
ax1.set_ylabel("Skill")
ax1.set_zlabel("Salary")
ax1.set_title("Degree-2 Polynomial Regression")


ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x1, x2, y, c='r', marker='o', label="Actual Salary")
ax2.plot_trisurf(x1, x2, y_pred3, alpha=0.5, color='gray')
ax2.set_xlabel("Age")
ax2.set_ylabel("Skill")
ax2.set_zlabel("Salary")
ax2.set_title("Degree-3 Polynomial Regression")

plt.tight_layout()
plt.show()
