mod load;
use load::*;

mod matrix;
use matrix::*;

struct NeuralNetwork {
    test_labels:  Vec<u8>,
    train_labels: Vec<u8>,
    test_data:  Vec<Vec<f64>>,
    train_data: Vec<Vec<f64>>,
    test_samples: i32,
    train_samples: i32,
    current_sample: usize, 
    train_accuracy: f32,
    test_accuracy: f32,
    epochs: i32,
    batch_size: i32,
    cost: f64,
    correct: i32,      // Number of predictions that were correct
    learn: f64,        // Learning rate
    h_bias: Matrix,    // Hidden layer biases
    h_weights: Matrix, // Hidden layer weights
    o_bias: Matrix,    // Output layer biases
    o_weights: Matrix, // Output layer weights
}

impl NeuralNetwork {
    fn new() -> NeuralNetwork {
        // Simple neural network with a 784 neuron input layer,
        // 100 neuron hidden layer and a 10 neuron output layer
        NeuralNetwork {
            cost: 0.0,
            epochs: 10,
            learn: 0.01,
            correct: 0,
            train_accuracy: 0.0,
            test_accuracy: 0.0,
            current_sample: 0,
            test_samples:  10000,
            train_samples: 60000,
            batch_size: 60000 / 600,
            h_weights:    Matrix::init(100, 784),
            h_bias:       Matrix::init(100, 1),
            o_weights:    Matrix::init(10, 100),
            o_bias:       Matrix::init(10, 1),
            test_data:    load_dataset("data/testing.idx3-ubyte"),
            test_labels:  load_label("data/testing-labels.idx1-ubyte"),
            train_data:   load_dataset("data/training.idx3-ubyte"),
            train_labels: load_label("data/training-labels.idx1-ubyte"),
        }
    }

    fn step(&mut self, input_layer: &Matrix, target_output: &Matrix) -> i32 {
        /* Forward propagation */
        let hidden = add(&dot(&self.h_weights, input_layer), &self.h_bias);
        let a_hidden = hidden.sigmoid(); // Activated hidden layer

        let output = add(&dot(&self.o_weights, &a_hidden), &self.o_bias);
        let mut a_output = output.softmax(); // Activated output layer

        /* Backward propagation */
        self.cost = a_output.mean_squared_error(target_output);
        let predicted = a_output.max();

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

        return predicted;
    }

    fn predict(&mut self, training: bool) {
        let mut img = Matrix::new(784, 1);
        if training {
            img.data = self.train_data[self.current_sample].clone();
        } else {
            img.data = self.test_data[self.current_sample].clone();
        }

        let mut img_label = Matrix::new(10, 1);
        let class = if training {
            self.train_labels[self.current_sample] as usize
        } else {
            self.test_labels[self.current_sample] as usize
        };
        img_label.data[class] = 1.0;

        let predicted = self.step(&img, &img_label);
        if predicted == class as i32 {
            self.correct += 1;
        }

        self.current_sample += 1;
    }

    fn train(&mut self) {
        println!("Training...");
        let batches = self.train_samples / self.batch_size;

        for _ in 0..self.epochs {
            for _ in 0..batches {
                for _ in 0..self.batch_size {
                    self.predict(true);
                }
            }
            self.current_sample = 0;
        }

        self.train_accuracy = (self.correct / self.train_samples) as f32;
    }

    fn test(&mut self) {
        println!("Testing...");
        self.correct = 0;
        self.current_sample = 0;

        for _ in 0..self.test_samples {
            self.predict(false);
        }

        self.test_accuracy = (self.correct / self.test_samples) as f32;
    }

    fn output_metrics(&self) {
        println!("{} cost", self.cost);
        println!("{}% testing accuracy", self.test_accuracy);
        println!("{}% training accuracy", self.train_accuracy);
        println!("{} epochs | batch size of {}", self.epochs, self.batch_size);
        println!("{} training samples | {} testing samples", self.train_samples, self.test_samples);
    }
}

fn main() {
    let mut nn = NeuralNetwork::new();
    nn.train();
    nn.test();
    nn.output_metrics();
}
