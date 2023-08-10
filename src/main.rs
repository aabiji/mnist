mod load;
mod matrix;
use load::*;
use matrix::*;

/*
Input layer: 784 neurons
Hidden layer: 100 neurons
Output layer: 10 neurons

The dot product between 100x784 and 784x1 results in 100x1. Then we can just add the bias, which happens to be 100x1!
The dot product between 10x100 and 100x1 results in 10x1. Then we can just add the bias, which happens to be 10x1!

Forward propagation:
hidden_layer           = dot(hidden_weights, input_layer) + hidden_bias
activated_hidden_layer = sigmoid(hidden_layer)
output_layer           = dot(output_weights, activated_hidden_layer) + output_bias
activated_output_layer = softmax(output_layer)

Backward propagation:
loss = mean_squared_error(target_output, activated_output_layer)

output_gradient         = activated_output_layer - target_output
output_weights_gradient = dot(output_gradient, transpose(activated_hidden_layer))

hidden_gradient         = dot(transpose(output_weights), output_gradient) * (activated_hidden_layer * (1 - activated_hidden_layer))
hidden_weights_gradient = dot(hidden_gradient, transpose(input_layer))

output_weights -= learning_rate * output_weights_gradient
output_bias    -= learning_rate * output_gradient
hidden_weights -= learning_rate * hidden_weights_gradient
hidden_bias    -= learning_rate * hidden_gradient
*/

struct NeuralNetwork {
    cost: f64,
    learn: f64,        // Learning rate
    h_bias: Matrix,    // Hidden layer biard
    h_weights: Matrix, // Hidden layer weights
    o_bias: Matrix,    // Output layer bias
    o_weights: Matrix, // Output layer weights
}

impl NeuralNetwork {
    fn new() -> NeuralNetwork {
        // Simple neural network with a 784 neuron input layer,
        // 100 neuron hidden layer and a 10 neuron output layer
        NeuralNetwork {
            cost: 0.0,
            learn: 0.01,
            h_weights: Matrix::new(100, 784),
            h_bias:    Matrix::new(100, 1),
            o_weights: Matrix::new(10, 100),
            o_bias:    Matrix::new(10, 1),
        }
    }

    fn train_step(&mut self) {
        let input_layer    = Matrix::new(784, 1);
        let target_output  = Matrix::new(10, 1);

        /* Forward propagation */
        let hidden = add(&dot(&self.h_weights, &input_layer), &self.h_bias);
        let a_hidden = hidden.sigmoid(); // Activated hidden layer

        let output = add(&dot(&self.o_weights, &a_hidden), &self.o_bias);
        let mut a_output = output.softmax(); // Activated output layer

        /* Backward propagation */
        self.cost = a_output.mean_squared_error(&target_output);

        // Output and output weights gradients
        let o_gradient = sub(&a_output, &target_output);
        let ow_gradient = dot(&o_gradient, &a_hidden.t());

        // Hidden and hidden weights gradients
        let h_gradient = mul(&dot(&self.o_weights.t(), &o_gradient), &a_hidden.sigmoidg());
        let hw_gradient = dot(&h_gradient, &input_layer.t());

        self.o_weights = sub(&self.o_weights, &scale(&ow_gradient, self.learn));
        self.o_bias    = sub(&self.o_bias, &scale(&o_gradient, self.learn));
        self.h_weights = sub(&self.h_weights, &scale(&hw_gradient, self.learn));
        self.h_bias    = sub(&self.h_bias, &scale(&h_gradient, self.learn));
    }
}

fn main() {
    let _test_data = load_dataset("data/testing.idx3-ubyte");
    let _test_labels = load_label("data/testing-labels.idx1-ubyte");

    let _train_data = load_dataset("data/training.idx3-ubyte");
    let _train_labels = load_label("data/training-labels.idx1-ubyte");

    let mut nn = NeuralNetwork::new();
    nn.train_step();
}
