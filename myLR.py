import numpy as np
import pandas as pd

class MyLR:
    
    def __init__(self, learning_rate=0.1, max_iterations=5000, tolerance=1e-6):
        # Initialising the class.
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):

        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, y_true, y_pred):
        # Avoiding log(0) by adding small epsilon
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y1 = y_true * np.log(y_pred)
        y2 = (1 - y_true) * np.log(1 - y_pred)
        cost = -np.mean(y1 + y2)
        return cost

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        """
        Normalizing features for better convergence. 
        Dint do it initially, and the results were horrible:
        [[ 0 565] [ 0 435]] 
        0.435 
        0.435 
        1.0 
        0.6062717770034843
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        X_normalized = (X - self.mean_) / self.std_
        
        n_samples, n_features = X_normalized.shape
        
        np.random.seed(42)  # For reproducibility
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Gradient descent
        prev_cost = float('inf')
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X_normalized, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost
            cost = self.cross_entropy_loss(y, predictions)
            self.cost_history.append(cost)
            
            dr = predictions - y
            # Compute gradients
            dw = (1/n_samples) * np.dot(X_normalized.T, dr)
            db = (1/n_samples) * np.sum(dr)
            

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
            prev_cost = cost
            
            # Print progress every 500 iterations
            if (i + 1) % 500 == 0:
                print(f"Iteration {i+1}, Cost: {cost:.6f}")
        
        return self


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Normalising the test data with the same mean and std as the training data
        X_normalized = (X - self.mean_) / self.std_
        y_hat = np.dot(X_normalized, self.weights) + self.bias
        probabilities =  self.sigmoid(y_hat)
        return (probabilities >= 0.5).astype(int)
    
