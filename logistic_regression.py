import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression  # For comparison (requires scikit-learn)
from sklearn.metrics import accuracy_score

# Generate synthetic binary classification dataset (500 samples, 5 features)
np.random.seed(42)  # For reproducibility
n = 500
n_per_class = n // 2
# Class 0: lower values
means_0 = [0, 0, 0, 0, 0]
cov = np.eye(5) * 2  # Increased variance for class overlap
X_0 = np.random.multivariate_normal(means_0, cov, n_per_class)

# Class 1: higher values
means_1 = [2, 2, 2, 2, 2]
X_1 = np.random.multivariate_normal(means_1, cov, n_per_class)

X = np.vstack((X_0, X_1))
y = np.hstack((np.zeros(n_per_class), np.ones(n_per_class)))

# Feature normalization
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Custom Logistic Regression class
class CustomLogisticRegression:
    def __init__(self, lr=0.1, epochs=10000, tol=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, X, y, theta):
        m = len(y)
        p = self.sigmoid(X @ theta)
        epsilon = 1e-10  # Avoid log(0)
        return -1/m * (np.dot(y, np.log(p + epsilon)) + np.dot((1 - y), np.log(1 - p + epsilon)))

    def grad(self, X, y, theta):
        m = len(y)
        p = self.sigmoid(X @ theta)
        return (1/m) * (X.T @ (p - y))

    def fit(self, X, y):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        self.theta = np.zeros(X_bias.shape[1])
        self.costs = [self.cost(X_bias, y, self.theta)]
        prev_cost = self.costs[0]
        self.iterations = 0
        for i in range(self.epochs):
            g = self.grad(X_bias, y, self.theta)
            self.theta -= self.lr * g
            current_cost = self.cost(X_bias, y, self.theta)
            self.costs.append(current_cost)
            self.iterations += 1
            if abs(current_cost - prev_cost) < self.tol:
                break
            prev_cost = current_cost
        return self.theta, self.costs

    def predict_proba(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(X_bias @ self.theta)

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

# Train custom model
custom_model = CustomLogisticRegression(lr=0.1, epochs=10000)
theta_custom, costs_custom = custom_model.fit(X_norm, y)

# Custom predictions and accuracy
preds_custom = custom_model.predict(X_norm)
acc_custom = np.mean(preds_custom == y)
print(f"Custom Model Accuracy: {acc_custom:.3f}")

# Train scikit-learn LogisticRegression for comparison
sklearn_model = LogisticRegression(penalty=None, fit_intercept=True, max_iter=10000)
sklearn_model.fit(X_norm, y)
theta_sklearn = np.hstack((sklearn_model.intercept_, sklearn_model.coef_.flatten()))
print("scikit-learn Theta:", theta_sklearn)

# scikit-learn predictions and accuracy
preds_sklearn = sklearn_model.predict(X_norm)
acc_sklearn = accuracy_score(y, preds_sklearn)
print(f"scikit-learn Accuracy: {acc_sklearn:.3f}")

# Compare coefficients
coeff_diff = theta_custom - theta_sklearn
print("Coefficient Differences:", coeff_diff)

# Plot cost history
plt.plot(costs_custom)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History During Training')
plt.show()

# Visualize decision boundary (project to first two features)
feature1, feature2 = 0, 1  # Select two key features
X_plot = X[:, [feature1, feature2]]
X_norm_plot = X_norm[:, [feature1, feature2]]

# Train a 2D custom model for visualization (using only two features)
custom_2d = CustomLogisticRegression()
custom_2d.fit(X_norm_plot, y)

# Plot data points
plt.scatter(X_plot[y==0, 0], X_plot[y==0, 1], color='red', label='Class 0')
plt.scatter(X_plot[y==1, 0], X_plot[y==1, 1], color='blue', label='Class 1')

# Decision boundary
theta_2d = custom_2d.theta
x1_plot = np.linspace(np.min(X_plot[:,0]), np.max(X_plot[:,0]), 100)
x1_norm = (x1_plot - X_mean[feature1]) / X_std[feature1]
x2_norm = -(theta_2d[0] + theta_2d[1] * x1_norm) / theta_2d[2]
x2_plot = x2_norm * X_std[feature2] + X_mean[feature2]
plt.plot(x1_plot, x2_plot, color='black', label='Decision Boundary')

plt.xlabel(f'Feature {feature1}')
plt.ylabel(f'Feature {feature2}')
plt.legend()
plt.title('2D Projection with Decision Boundary')
plt.show()
