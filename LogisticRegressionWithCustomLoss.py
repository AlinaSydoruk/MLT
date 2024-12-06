import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import expit  # Sigmoid function


class LogisticRegressionWithCustomLoss:
    def __init__(self, loss_function='logistic', learning_rate=0.01, epochs=1000):
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def sigmoid(self, z):
        return expit(z)  # Sigmoid function

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights) + self.bias) > 0.5).astype(int)

    def logistic_loss(self, y, y_pred):
        return np.mean(-y * np.log(y_pred + 1e-10) - (1 - y) * np.log(1 - y_pred + 1e-10))

    def adaboost_loss(self, y, y_pred):
        # AdaBoost loss: L = exp(-y * f(x))
        return np.mean(np.exp(-y * y_pred))

    def binary_crossentropy_loss(self, y, y_pred):
        # Binary Crossentropy: L = - [y log(p) + (1-y) log(1-p)]
        return np.mean(-y * np.log(y_pred + 1e-10) - (1 - y) * np.log(1 - y_pred + 1e-10))

    def fit(self, X_train, y_train, X_test, y_test):
        m, n = X_train.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for epoch in range(self.epochs):
            # Forward pass
            y_train_pred = self.sigmoid(np.dot(X_train, self.weights) + self.bias)
            y_test_pred = self.sigmoid(np.dot(X_test, self.weights) + self.bias)

            # Compute the loss
            if self.loss_function == 'logistic':
                train_loss = self.logistic_loss(y_train, y_train_pred)
                test_loss = self.logistic_loss(y_test, y_test_pred)
            elif self.loss_function == 'adaboost':
                train_loss = self.adaboost_loss(y_train, y_train_pred)
                test_loss = self.adaboost_loss(y_test, y_test_pred)
            elif self.loss_function == 'binary_crossentropy':
                train_loss = self.binary_crossentropy_loss(y_train, y_train_pred)
                test_loss = self.binary_crossentropy_loss(y_test, y_test_pred)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            # Calculate gradients
            error = y_train_pred - y_train
            dw = np.dot(X_train.T, error) / m
            db = np.sum(error) / m

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track accuracy
            train_accuracy = accuracy_score(y_train, self.predict(X_train))
            test_accuracy = accuracy_score(y_test, self.predict(X_test))
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    def plot_learning_curves(self):
        plt.figure(figsize=(12, 6))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.test_losses, label='Test Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves ({self.loss_function})')
        plt.legend()

        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(self.test_accuracies, label='Test Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves ({self.loss_function})')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy (Test): {accuracy * 100:.2f}%')
        return accuracy
