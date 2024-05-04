import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class Module():
    def __init__(self):
        self.prev = None 
        self.output = None 

    learning_rate = 1E-2

    def __call__(self, input):
        if isinstance(input, Module):
            self.prev = input
            self.output = self.forward(input.output)
        else:
            self.output = self.forward(input)
                        
        return self

    def forward(self, *input):
        raise NotImplementedError

    def backwards(self, *input):
        raise NotImplementedError


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        self.output = 1/(1 + np.exp(-input))
        return self.output

    def backwards(self, gradient):
        der = self.output * (1 - self.output)
        return gradient * der


class Linear(Module):
    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        self.W = np.random.randn(output_size, input_size)
        self.b = np.zeros((output_size, 1))
        self.input = None
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def forward(self, input): 
        self.input = input
        self.output = np.dot(input, self.W.T) + self.b.reshape(1, -1)
        return self.output

    def backwards(self, gradient):
        nextgrad = np.dot(gradient, self.W)
        
        dW = np.dot(gradient.T, self.input)
        db = np.sum(gradient, axis=0, keepdims=True).reshape(self.b.shape)

        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * dW**2
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * db**2

        self.W -= self.learning_rate * self.m_W / (np.sqrt(self.v_W) + self.epsilon)
        self.b -= self.learning_rate * self.m_b / (np.sqrt(self.v_b) + self.epsilon)

        return nextgrad


class Loss:
    def __init__(self):
        self.prev = None

    def __call__(self, input):
        self.prev = input
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError


class MeanErrorLoss(Loss):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()
        self.labels = None
        self.input = None

    def forward(self, input, labels): 
        self.input = input
        self.labels = labels
        self.output = 0.5 * np.mean((input - labels)**2)
        return self.output

    def backwards(self):
        return 2 * (self.input - self.labels) / self.input.size


class Network(Module):
    def __init__(self, layers):
        super(Network, self).__init__()
        self.layers = layers
        
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def backwards(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backwards(grad)
        return grad

    def predict(self, data):
        output = self.forward(data)
        return output.output

    def accuracy(self, test_data, test_labels):
        predictions = self.predict(test_data)
        return np.mean(predictions == test_labels)


def train(model, data, labels, num_iterations, minibatch_size, learning_rate):
    n = len(data)
    for iteration in range(num_iterations):
        permutation = np.random.permutation(n)
        data = data[permutation]
        labels = labels[permutation]
        tot_loss = 0
        for i in range(0, n, minibatch_size):
            minibatch_data = data[i:i+minibatch_size]
            minibatch_labels = labels[i:i+minibatch_size]
            output = model.forward(minibatch_data)
            loss = MeanErrorLoss()
            loss_value = loss.forward(output.output, minibatch_labels)
            tot_loss += loss_value
            grad = loss.backwards()
            model.backwards(grad)
        print("Iteration:", iteration)
        print("Mean Squared Error:", tot_loss)
            
