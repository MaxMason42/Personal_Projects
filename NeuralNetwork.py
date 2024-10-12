import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.where(x < -100, 0, x)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(z):
    return z * (1.0 - z)

class Layer:
    def __init__(self, weights, bias, outputs, deltas) -> None: 
        self.weights = weights
        self.bias = bias
        self.outputs = outputs
        self.deltas = deltas


class NeuralNetwork:
    def __init__(self, layers = None, learning_rate =1E-2):
        self.layers = [Layer(weights=np.random.rand(layers[i], layers[i - 1]), bias=np.random.rand(layers[i]),
                             outputs=np.zeros(layers[i]), deltas=np.zeros(layers[i]),)
                       for i in range(1, len(layers))]
        
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = [np.zeros_like(layer.weights) for layer in self.layers]
        print(self.m[0])
        self.v = [np.zeros_like(layer.weights) for layer in self.layers]
    
    def outputs(self):
        return self.layers[-1].outputs
    
    def forward(self, input):
        for layer in self.layers:
            layer.outputs = sigmoid(np.dot(layer.weights, input) + layer.bias)
            input = layer.outputs
        return self.layers[-1].outputs
    
    def backprop(self, inputs, labels, t):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                layer.deltas = (layer.outputs - labels) * sigmoid_derivative(layer.outputs)
            else:
                next_layer = self.layers[i + 1]
                layer.deltas = np.dot(next_layer.weights.T, next_layer.deltas) * sigmoid_derivative(layer.outputs)
        self.update_weights(inputs, t)

    def update_weights(self, inputs, t):
        '''
        for i in range(len(self.layers)):
            
            layer = self.layers[i]
            previous_layer_outputs = self.layers[i - 1].outputs if i > 0 else inputs
            layer.weights -= (np.dot(layer.deltas[np.newaxis].T, previous_layer_outputs[np.newaxis]) * self.learning_rate)
            layer.bias -= layer.deltas * self.learning_rate
            '''
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer_input = np.atleast_2d(inputs)
            else:
                layer_input = np.atleast_2d(self.layers[i - 1].outputs)
            delta = np.atleast_2d(layer.deltas)

            # Compute gradients
            grad = np.dot(layer_input.T, delta).T
            
            #grad2 = np.dot(layer_input.T, delta).T
            
            #print(grad)
            #print(grad2)

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grad)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - np.power(self.beta1, t))

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - np.power(self.beta2, t))

            # Update weights
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def train(self, inputs, labels, epochs, minibatch_size, learning_rate = 1E-2):
        n = len(inputs)
        self.learning_rate = learning_rate
        t = 1
        for epoch in range(epochs):
            sum_error = 0.0

            # Shuffle the training data
            permutation = np.random.permutation(n)
            inputs = inputs[permutation]
            labels = labels[permutation]

            # Split the data into mini-batches
            for i in range(0, n, minibatch_size):
                mini_inputs = inputs[i:i+minibatch_size]
                mini_labels = labels[i:i+minibatch_size]

                # Perform forward and backward pass for each mini-batch
                for j, row in enumerate(mini_inputs):
                    actual = self.forward(row)
                    self.backprop(row, mini_labels[j], t)
                    sum_error += self.mse(actual, mini_labels[j])
                    t += 1

            print(f"Mean squared error: {sum_error / minibatch_size}")
            print(f"epoch={epoch}")
    
    
    def mse(self, actual, labels) :
        return np.power(actual - labels, 2).mean()
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        return np.argmax(outputs)
    
    
def test_make_prediction_with_network():
    dataset = np.array(
        [
            [2.7810836, 2.550537003],
            [1.465489372, 2.362125076],
            [3.396561688, 4.400293529],
            [1.38807019, 1.850220317],
            [3.06407232, 3.005305973],
            [7.627531214, 2.759262235],
            [5.332441248, 2.088626775],
            [6.922596716, 1.77106367],
            [8.675418651, -0.242068655],
            [7.673756466, 3.508563011],
        ]
    )
    expected = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    # 2 input neurons, 3 hidden neurons, 2 output neurons
    network = NeuralNetwork([len(dataset[0]), 3, len(expected[0])])
    network.train(dataset, expected, 40, 5, 0.5)
    for i in range(len(dataset)):
        prediction = network.predict(dataset[i])
        print(
            f"{i} - Expected={np.where(expected[i] == expected[i].max())[0][0]}, Got={prediction}"
        )

def test_quadratic():
    # Define the quadratic function
    def quadratic(x):
        return x**3

    # Generate some training data
# Generate some data
    x_data = np.linspace(-4, 9, 1000)
    y_data = quadratic(x_data)

    # Choose a threshold for classification
    threshold = 36

    # Classify the data
    labels = np.where(y_data > threshold, 1, 0)

    # Reshape the data for the neural network
    x_data = x_data.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    # Initialize the neural network
    nn = NeuralNetwork([1, 10, 5, 1])

    # Train the neural network
    nn.train(x_data, labels, epochs=200, minibatch_size=50, learning_rate=0.01)

    # Test the neural network
    
    #print(labels)

    # Use the neural network to predict the test data
    print(sum(labels))
    count = 0
    for i in range(len(x_data)):
        prediction = nn.predict(x_data[i])
        #print(
        #    f"{i} - Expected={labels[i]}, Got={prediction}"
        #)
        if prediction == labels[i]:
            count += 1
    print("Number of correct predictions:", count)

def test_image():
    
    data = scipy.io.loadmat("C:/Users/Maxim/Downloads/nn_data.mat")
    
    X1, X2 = data['X1'], data['X2']
    Y1, Y2 = data['Y1'], data['Y2']


    width = int(np.max(X1[:, 0])) + 1
    height = int(np.max(X1[:, 1])) + 1

    # Create an empty 2D array for the image
    image = np.zeros((width, height))

    # Fill the image array with the color values from Y1
    for i in range(len(X1)):
        x = int(X1[i, 0])
        y = int(X1[i, 1])
        image[x, y] = Y1[i]

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.show()
    
    width2 = int(np.max(X2[:, 0])) + 1
    height2 = int(np.max(X2[:, 1])) + 1

    # Create an empty 2D array for the image
    image2 = np.zeros((width2, height2, 3))
    
    print(Y2)

    # Fill the image array with the color values from Y1
    for i in range(len(X2)):
        x2 = int(X2[i, 0])
        y2 = int(X2[i, 1])
        image2[x2, y2] = Y2[i] / 255

    # Display the image
    plt.imshow(image2)
    plt.show()

    nn1 = NeuralNetwork([X1.shape[1], 225, Y1.shape[1]])
    
    X1_norm = X1 / X1.shape[1]
    Y1_norm = Y1 / 255
    
    nn1.train(X1_norm, Y1_norm, epochs=1000, minibatch_size=32, learning_rate=1000)


if __name__ == "__main__":
    #test_make_prediction_with_network()
    #test_quadratic()
    test_image()
