import streamlit as st
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for each layer
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def forward(self, inputs):
        # Forward pass through the network
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output

    def backward(self, inputs, targets, learning_rate):
        # Backpropagation
        output_error = targets - self.output
        output_delta = output_error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
        self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, learning_rate, epochs):
        # Train the neural network
        for epoch in range(epochs):
            self.forward(inputs)
            self.backward(inputs, targets, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - self.output))
                st.write(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, inputs):
        # Make predictions
        return self.forward(inputs)

def main():
    st.title("Neural Network Streamlit App")

    st.sidebar.header("Model Configuration")
    input_size = st.sidebar.number_input("Input size:", min_value=1, value=2)
    hidden_size = st.sidebar.number_input("Hidden size:", min_value=1, value=3)
    output_size = st.sidebar.number_input("Output size:", min_value=1, value=1)

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    st.sidebar.header("Training Configuration")
    learning_rate = st.sidebar.number_input("Learning rate:", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
    epochs = st.sidebar.number_input("Epochs:", min_value=1, value=1000)

    st.subheader("Training Data")
    inputs = []
    targets = []
    for i in range(input_size):
        input_data = []
        for j in range(input_size):
            input_value = st.number_input(f"Input {i+1}, {j+1}", value=0)
            input_data.append(input_value)
        inputs.append(input_data)
    
    for i in range(output_size):
        target_data = []
        for j in range(output_size):
            target_value = st.number_input(f"Target {i+1}, {j+1}", value=0)
            target_data.append(target_value)
        targets.append(target_data)

    inputs = np.array(inputs)
    targets = np.array(targets)

    if st.button("Train"):
        nn.train(inputs, targets, learning_rate, epochs)
        st.success("Training completed!")

        st.subheader("Predictions:")
        predictions = nn.predict(inputs)
        st.write(predictions)

if __name__ == "__main__":
    main()
