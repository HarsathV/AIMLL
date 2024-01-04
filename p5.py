import numpy as np

# X = (Hours sleeping, Hours Studying), y = (test score of the student)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X / np.amax(X, axis=0)  # maximum of X array
y = y / 100  # maximum test score is 100

class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3
        self.w1 = np.random.rand(self.input_size, self.hidden_size)  # (3x2) weight
        self.w2 = np.random.rand(self.hidden_size, self.output_size)  # (3x1) weight

    def feedforward(self, X):
        # forward propagation through the network
        self.z = np.dot(X, self.w1)  # dot product of X(input) & w1
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.w2)  # dot product of z2 and w2
        output = self.sigmoid(self.z3)
        return output

    def sigmoid(self, s, deriv=False):
        if deriv:
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def backward(self, X, y, output):
        # backward propagate through the network
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        self.z2_error = self.output_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)
        self.w1 += X.T.dot(self.z2_delta)
        self.w2 += self.z2.T.dot(self.output_delta)

    def train(self, X, y):
        output = self.feedforward(X)
        self.backward(X, y, output)

NN = NeuralNetwork()

for i in range(50000):
    if i % 100 == 0:
        print("Loss: " + str(np.mean(np.square(y - NN.feedforward(X)))))
    NN.train(X, y)

print("Input: " + str(X))
print("Actual output: " + str(y))
print("Predicted output: " + str(NN.feedforward(X)))
print("Loss: " + str(np.mean(np.square(y - NN.feedforward(X)))))
