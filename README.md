# Neural Network from Scratch (NumPy, MNIST)

This project implements a two-layer feedforward neural network from scratch using only NumPy. It’s designed for learners who want to deepen their understanding of the math and code behind forward/backward propagation, softmax cross‑entropy, and gradient descent—without high‑level ML libraries.

—  
## 🧠 Project Highlights

- Built entirely from scratch (NumPy + Python)
- Trains on the MNIST handwritten digits dataset
- Vectorized forward/backward propagation
- ReLU (hidden) + Softmax (output)
- Mini‑batch gradient descent with shuffling
- Input normalization and one‑hot labels
- Achieves >96% accuracy on MNIST (dev)

—  
## 📚 Prerequisites

You should be comfortable with:
- Linear algebra and matrix operations
- The affine transform y = Wx + b
- Derivatives and the chain rule
- Basics of neural nets (layers, activations, loss)

—  
## ⚙️ Network Architecture

- Input: 784 features (28×28 grayscale image flattened)
- Hidden: 128 neurons with ReLU (configurable)
- Output: 10 neurons with Softmax for class probabilities

—  
## 🔁 Training Workflow

1) Data prep
- Load MNIST (via kagglehub in Kaggle/Colab)
- Normalize pixels to [0,1]
- One‑hot encode labels to shape (10, m)

2) Forward propagation
- Z1 = W1·X + b1
- A1 = ReLU(Z1)
- Z2 = W2·A1 + b2
- A2 = softmax(Z2)

3) Loss (cross‑entropy)
- L = −(1/m) Σ Y_onehot · log(A2)

4) Backpropagation
- dZ2 = A2 − Y_onehot
- dW2 = (dZ2 · A1ᵀ)/m, db2 = mean(dZ2)
- dA1 = W2ᵀ · dZ2
- dZ1 = dA1 ⊙ ReLU’(Z1)
- dW1 = (dZ1 · Xᵀ)/m, db1 = mean(dZ1)

5) Update
- W ← W − α·dW
- b ← b − α·db

6) Mini‑batch loop
- Shuffle indices each epoch
- Iterate batches (e.g., 128)
- Track epoch loss and accuracy (optionally dev accuracy)

—  
## 🔢 Core Functions (Snippets)

```python
def relu(z):
    return np.maximum(z, 0)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)  # stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

```python
def back_prop(w1, b1, w2, b2, z1, a1, z2, a2, X, Y_onehot):
    m  = X.shape[1]
    dz2 = a2 - Y_onehot
    dw2 = (dz2 @ a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    da1 = w2.T @ dz2
    dz1 = da1 * (z1 > 0)
    dw1 = (dz1 @ X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    return dw1, db1, dw2, db2
```

—  
## 📈 Results

- Reached >96% accuracy on MNIST (dev) with:
  - Hidden=128, batch_size=128, lr=0.1, epochs=10
- Further gains possible with:
  - More hidden units or layers
  - Learning‑rate schedules
  - Optimizers (Adam/RMSProp)
  - Regularization (L2/dropout)
  - Early stopping

—  
## 🚀 Getting Started

1) Environment
- Kaggle or Colab recommended
- Dataset via kagglehub (project includes example)

2) Train (mini‑batch)
```python
w1, b1, w2, b2 = train_minibatch(
    X=x_train, y=y_train,
    epochs=10, batch_size=128, lr=0.1,
    X_val=x_dev, y_val=y_dev, seed=42
)
```

3) Evaluate
```python
_, _, _, a2_tr = forward_prop(w1, b1, w2, b2, x_train)
print("Train acc:", (predictions(a2_tr) == y_train).mean())

_, _, _, a2_val = forward_prop(w1, b1, w2, b2, x_dev)
print("Dev acc:", (predictions(a2_val) == y_dev).mean())
```

—  
## 🔍 Predict a Single Image

```python
idx = 123  # any index in the dev set
x_sample = x_dev[:, idx:idx+1]  # keep shape (784,1)
y_true = int(y_dev[idx])

_, _, _, a2 = forward_prop(w1, b1, w2, b2, x_sample)
y_pred = int(predictions(a2)[0])
print(f"Pred: {y_pred}, True: {y_true}, Correct? {y_pred == y_true}")
```

Optional visualization:
```python
plt.imshow(x_sample.reshape(28, 28), cmap='gray')
plt.title(f"Pred: {y_pred} | True: {y_true}")
plt.axis('off'); plt.show()
```

—  
## 📁 File Structure

- neural_network_from_scratch.py — full training and inference code
- Dataset — pulled via kagglehub (see notebook/script)

—  
## 🧠 Future Improvements

- MLflow logging (per‑epoch loss/accuracy, artifacts)
- Adam optimizer and LR schedulers
- Regularization (L2, dropout)
- Early stopping and better evaluation splits
- Export weights and simple inference CLI or web demo

—  
## 👨‍🔬 Author

Built by [Salmanul Faris](https://github.com/SalmanFaris7) as a hands‑on exploration of neural networks from first principles.
