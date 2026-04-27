# Neural Network from Scratch - Presentation Documentation

## Overview

This project demonstrates a neural network implemented entirely from scratch using NumPy, without relying on deep learning frameworks like TensorFlow or PyTorch. The implementation covers the fundamental concepts of neural networks including forward propagation and various activation functions.

## Architecture

### Network Structure
- **Input Layer**: 16 neurons
- **Hidden Layer 1**: 10 neurons
- **Hidden Layer 2**: 10 neurons  
- **Output Layer**: 1 neuron

### Forward Propagation Process

```
Input (A_0) → Hidden Layer 1 → Hidden Layer 2 → Output (Y)
```

1. **Input Layer (A_0)**: Receives input data with 16 features

2. **Hidden Layer 1**:
   - Z[1] = A[0] × W[1] + B[1]
   - Activation: **ReLU** (Rectified Linear Unit)
   - Formula: A[1] = max(0, Z[1])
   - ReLU returns 1 for positive values, 0 otherwise

3. **Hidden Layer 2**:
   - Z[2] = A[1] × W[2] + B[2]
   - Activation: **Sigmoid**
   - Formula: A[2] = 1 / (1 + e^(-z))
   - Squashes values between 0 and 1

4. **Output Layer**:
   - Y = A[2] × W[3] + B[3]
   - Final prediction

## Activation Functions

### ReLU (Rectified Linear Unit)
- **Purpose**: Introduces non-linearity in hidden layers
- **Formula**: f(x) = max(0, x)
- **Advantages**:
  - Computationally efficient
  - Reduces vanishing gradient problem
  - Accelerates convergence

### Sigmoid
- **Purpose**: Used for binary classification output
- **Formula**: f(x) = 1 / (1 + e^(-x))
- **Output Range**: (0, 1)
- **Use Case**: Probability estimation

### Softmax
- **Purpose**: Multi-class classification
- **Formula**: f(x_i) = e^(x_i) / Σ e^(x_j)
- **Output**: Probability distribution summing to 1

## Key Formulas

### Matrix Multiplication
```
Z = A × W + b
```
- A: Activation from previous layer
- W: Weight matrix
- b: Bias vector
- Z: Pre-activation value

### Dimensions
- Input: (1, 16)
- Weight 1: (16, 10)
- Weight 2: (10, 10)
- Weight 3: (10, 1)

## Visualizations

See `imgs/` directory for:
- `neural_network.png` - Network architecture diagram
- `activation_function.png` - Activation functions visualization
- `relu.jpg` - ReLU function plot
- `softmax.png` - Softmax function visualization

## MNIST Dataset

The project includes MNIST handwritten digit classification:
- Training data: `mnist_train.csv`
- 28×28 pixel images (784 features)
- 10 classes (digits 0-9)
- Saved model: `model.npz`

## Requirements

- Python 3.x
- NumPy
- Pandas

## Usage

Open `MyMain.ipynb` in Jupyter Notebook to run the neural network implementation and training code.

## Learning Outcomes

After this presentation, you will understand:
1. How neural networks perform forward propagation
2. The role of activation functions (ReLU, Sigmoid, Softmax)
3. How to implement a simple neural network from scratch
4. Matrix operations in neural networks
5. Weight and bias initialization

## References

- Deep Learning Specialization courses
- Neural Networks and Deep Learning documentation