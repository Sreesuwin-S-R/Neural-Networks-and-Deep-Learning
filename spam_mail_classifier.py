import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FeedForward2HL:
    def __init__(self):
        self.W1 = np.array([0.5, -0.2, 0.3])
        self.W2 = np.array([0.4, 0.1, -0.5])
        self.Wout = 1  

    def forward(self, x):
        # Hidden Layer 1 → ReLU
        z1 = np.sum(x * self.W1)
        h1 = relu(z1)

        # Hidden Layer 2 → Sigmoid
        z2 = h1 * np.sum(self.W2)
        h2 = sigmoid(z2)

        # Output → Sigmoid
        z3 = h2 * self.Wout
        output = sigmoid(z3)
        prob = output * 100
        message = f'The mail is {prob:.2f}% likely to be spam.'

        return message

x = np.array([1, 0, 1])
model = FeedForward2HL()
print(model.forward(x))
