<img src="./imgs/from scratch.jpg">

# Neural Network from Scratch

> Yeah, you read that right. I do this from scratch. In this documentation, I'm going to build a tiny neural network in pure Python. No libraries. No shortcuts. No TensorFlow. No PyTorch. **Just me and math.**

---

## Prerequisites

Before we start, you should know:

- The basics of what a neural network is (layers, weights, biases, activation functions)
- Some Python experience helps, but I'll explain things as we go

---

## Table of Contents

1. [Neural Network Architecture](#neural-network-architecture)
2. [Some Important Concepts](#some-important-concepts)
3. [Our Case with the MNIST Dataset](#our-case-with-the-mnist-dataset)
4. [Full Code Listing](#full-code-listing)
5. [Building and Running](#building-and-running)

---

## 1. Neural Network Architecture

A neural network is a model that tries to simulate how the human brain works. In the human brain, we have something called a **neuron**.

> **By the way** — a neuron is a COMPLEX FUNCTION. This function can learn new patterns.

This network contains three layers:

| Layer | Role |
|---|---|
| **Input Layer** | Accepts the raw inputs (numbers, images, audio) |
| **Hidden Layer** | Learns patterns from the data |
| **Output Layer** | Produces the prediction (temperature) or classification(cat/dog) |

<img src="./imgs/neural_network.png" width="650" alt="Neural Network Diagram" />

### The Three Layers Explained

**Input Layer**
The layer that accepts inputs — for example, temperature, image, or audio. But all these inputs are numbers.

**Hidden Layer**
The layer that learns from the data.

**Output Layer**
The output depends on what we want to predict. For example:
- Is this a cat or a dog?
- Is this email spam or not?

---

## 2. Some Important Concepts

### How many neurons in each layer?

**Input Layer**

It depends on the inputs. If it's an image, the number of neurons equals the image resolution — for example, 28 × 28 = 784 neurons.

**Hidden Layer**

There's no strict rule here, but keep this in mind:

> More hidden layers is better than more neurons in one hidden layer. As Internet said 1–2 layers solve most common problems.

**Output Layer**

It depends on what we want in output. For example, if we're classifying 10 digits (0–9), we need 10 output neurons. etc...


### Activation Functions

In broad terms, activation functions are necessary to prevent linearity. Without them, the data would pass through the nodes and layers of the network only going through linear functions (a*x+b). The composition of these linear functions is again a linear function and so no matter how many layers the data goes through, the output is always the result of a linear function. An example is explained in the diagram below.

<img src="./imgs/activation_function.png">

An example showing the benefits of nonlinear functions for fitting data models.


<img src="./imgs/benfit_of_activation_function.png">

FROM: https://towardsdatascience.com/the-importance-and-reasoning-behind-activation-functions-4dc00e74db41/

---

#### ReLU — Activation Function 

For the hidden layer, we use **ReLU (Rectified Linear Unit)**:

$$
f(x) = \max(0, x)
$$

<img src="./imgs/relu.jpg" width="500" alt="ReLU Activation Function" />

**How it works:**

| Condition | Output |
|---|---|
| x > 0 | f(x) = x |
| x ≤ 0 | f(x) = 0 |

Simple idea: if the value is negative, kill it means return 0. If it's positive, keep it. means return it.

---

#### Softmax — Activation Function

For the output layer, we use **Softmax**:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

<img src="./imgs/softmax.png" width="500" alt="Softmax Activation Function" />

Softmax turns the output into **probabilities** — every output value is between 0 and 1, and they all add up to 1. This makes it perfect for classification.

---

> **NOTE:** We apply **ReLU** to each neuron in the hidden layer, and **Softmax** to each neuron in the output layer.

---

### Cross Entropy Loss Function
is function mesure how close a model’s predictions are to the correct answers in **classification** problems.

and Dependin in Problem There a Type of Cross Entropy Function:

#### Binary Cross-Entropy Loss 
is a widely used loss function in binary classification problems. For a dataset with N instances, the Binary Cross-Entropy Loss is calculated as:

$$
\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(p_i) + (1 - y_i)\log(1 - p_i) \right)
$$

#### Multiclass Cross Entropy Loss
as categorical cross-entropy or softmax loss is a widely used loss function for training models in multiclass classification problems. For a dataset with N instances, Multiclass Cross-Entropy Loss is calculated as

$$
\text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} \left( y_{i,j} \cdot \log(p_{i,j}) \right)
$$

#### Where:

- **N**: number of samples  
- **C**: number of classes  
- **y_{i,j}**: equals 1 if class *j* is the correct label for sample *i*, otherwise 0  
- **p_{i,j}**: model-predicted probability that sample *i* belongs to class *j*  

> In Forward Padd We Gonna use this Function to See Prediction of Our model and Then Calculate Accuracy.


GOOD RESSOURCE: https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/


## Some Here
### Our Architecture

For this project, we're going to use:

```
Input Layer  →  Hidden Layer  →  Output Layer
```

<img src="./imgs/neural_network1.png" width="650" alt="Our Architecture" />

---
