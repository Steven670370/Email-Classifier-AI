import random
import math

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim=1, lr=0.1):
        self.lr = lr

        # weights
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]

        self.W2 = [random.uniform(-0.1, 0.1) for _ in range(hidden_dim)]
        self.b2 = 0.0

    # ---------- activations ----------
    def relu(self, x):
        return max(0.0, x)

    def relu_grad(self, x):
        return 1.0 if x > 0 else 0.0

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # ---------- forward ----------
    def forward(self, x):
        self.z1 = []
        self.h = []

        for i in range(len(self.W1)):
            z = sum(w * xi for w, xi in zip(self.W1[i], x)) + self.b1[i]
            self.z1.append(z)
            self.h.append(self.relu(z))

        self.z2 = sum(w * hi for w, hi in zip(self.W2, self.h)) + self.b2
        self.out = self.sigmoid(self.z2)

        return self.out

    # ---------- backward ----------
    def backward(self, x, y):
        # output gradient (MSE loss)
        d_out = self.out - y

        # W2, b2
        for i in range(len(self.W2)):
            self.W2[i] -= self.lr * d_out * self.h[i]
        self.b2 -= self.lr * d_out

        # hidden layer
        for i in range(len(self.W1)):
            dh = d_out * self.W2[i] * self.relu_grad(self.z1[i])
            for j in range(len(self.W1[i])):
                self.W1[i][j] -= self.lr * dh * x[j]
            self.b1[i] -= self.lr * dh

    def train_step(self, x, y):
        pred = self.forward(x)
        self.backward(x, y)
        return (pred - y) ** 2