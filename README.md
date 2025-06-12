# Neural Network from Scratch

This project implements a **two-layer neural network** from scratch using only NumPy and standard Python. It is designed for those who already have a foundational understanding of how neural networks work and want to explore the **mathematical and programmatic inner workings of backpropagation** and gradient descent.

---

## 🧠 Project Highlights

* **Built from the ground up** with no high-level machine learning libraries.
* Trained on the **MNIST digit classification dataset**.
* Implements:

  * ReLU and Softmax activation functions
  * Forward and backward propagation
  * Batch gradient descent with optional learning rate decay

---

## 📚 Prerequisites

You should already be familiar with:

* Matrix operations and linear algebra basics
* The equation `y = wx + b`
* Derivatives and chain rule
* Neural network architecture and flow

---

## ⚙️ Network Architecture

* **Input Layer**: 784 features (28×28 pixel images)
* **Hidden Layer**: 10 neurons with ReLU activation
* **Output Layer**: 10 neurons with Softmax for classification

---

## 🔁 Training Workflow

1. **Initialization**: Random weights and zero biases for both layers.
2. **Forward Propagation**:

   * `Z1 = W1·X + b1`
   * `A1 = ReLU(Z1)`
   * `Z2 = W2·A1 + b2`
   * `A2 = Softmax(Z2)`
3. **Backpropagation**:

   * Compute `dZ2 = A2 - Y_one_hot`
   * Chain derivatives to update gradients: `dW`, `db`
4. **Parameter Update**:

   * `W -= α * dW`
   * `b -= α * db`
   * Includes optional **step decay** for learning rate

---

## 🔢 Sample Code Snippets

```python
def relu(z):
    return np.maximum(z, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

```python
def back_prop(...):
    dz2 = a2 - Y
    dw2 = dz2 @ a1.T / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = (w2.T @ dz2) * (z1 > 0)
    dw1 = dz1 @ X.T / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
```

---

## 📈 Accuracy & Optimization

* Achieved \~79% accuracy on MNIST subset using simple batch gradient descent.
* Includes room for experimenting with:

  * Learning rate decay
  * More advanced optimizers (Adam, RMSprop)
  * Additional layers or neuron counts

---

## 🚀 Getting Started

1. Clone the repo and open in Colab or your preferred IDE.
2. Make sure `kagglehub` and dataset imports are correctly configured.
3. Run `gradient_decent()` to train the model.
4. Test predictions and visualize results.

---

## 🧪 Try It

```python
x_sample = x_train[:, 0]
_, _, _, a2 = foward_prop(w1, b1, w2, b2, x_sample)
print("Prediction:", predictions(a2))
```

---

## 📁 File Structure

* `neural_network_from_scratch.py` – Full training and inference code
* `MNIST` dataset is pulled automatically using `kagglehub`

---

## 🧠 Future Improvements

* Add mini-batch gradient descent
* Implement regularization
* Export model for inference use
* GUI or web demo

---

## 👨‍🔬 Author

Built by [Salmanul Faris](https://github.com/faris71) as a deep learning exploration project.

