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
        epsilon = 1e-10
        return -1/m * (np.dot(y, np.log(p + epsilon)) + np.dot((1 - y), np.log(1 - p + epsilon)))

    def grad(self, X, y, theta):
        m = len(y)
        p = self.sigmoid(X @ theta)
        return (1/m) * (X.T @ (p - y))

    def fit(self, X, y):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        self.theta = np.zeros(X_bias.shape[1])
        self.costs = [self.cost(X_bias, y, self.theta)]
        prev_cost = self.costs[0]

        for i in range(self.epochs):
            g = self.grad(X_bias, y, self.theta)
            self.theta -= self.lr * g
            current_cost = self.cost(X_bias, y, self.theta)
            self.costs.append(current_cost)

            if abs(current_cost - prev_cost) < self.tol:
                break
            prev_cost = current_cost

        return self.theta, self.costs

    def predict_proba(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(X_bias @ self.theta)

    def predict(self, X):
        return self.predict_proba(X) >= 0.5
