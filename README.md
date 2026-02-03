# MNIST Neural Network from Scratch

A simple, from-scratch implementation of a **two-layer Feed-Forward Neural Network** using only `NumPy`. This project demonstrates the core mathematics of deep learning by classifying handwritten digits from the MNIST dataset without using high-level libraries like TensorFlow or PyTorch.

---

## 🏗️ Network Architecture

The network follows a standard dense architecture:
* **Input Layer:** 784 neurons (representing each pixel of a $28 \times 28$ image).
* **Hidden Layer:** 10 neurons with **ReLU** activation.
* **Output Layer:** 10 neurons (digits 0-9) with **Softmax** activation.



---

## 🔢 Mathematical Breakdown

### 1. Forward Propagation
Forward propagation calculates the output by passing the input through each layer.

**Hidden Layer:**
$$Z_1 = W_1 \cdot X + b_1$$
$$A_1 = \text{ReLU}(Z_1)$$

**Output Layer:**
$$Z_2 = W_2 \cdot A_1 + b_2$$
$$A_2 = \text{Softmax}(Z_2)$$

### 2. Backward Propagation
Backprop computes the gradient of the loss function with respect to weights and biases to "teach" the network.

**Output Error:**
$$dZ_2 = A_2 - Y_{one\_hot}$$

**Weight/Bias Gradients (Layer 2):**
$$dW_2 = \frac{1}{m} dZ_2 \cdot A_1^T$$
$$db_2 = \frac{1}{m} \sum dZ_2$$

**Hidden Layer Error:**
$$dZ_1 = W_2^T \cdot dZ_2 \odot \text{ReLU}'(Z_1)$$

**Weight/Bias Gradients (Layer 1):**
$$dW_1 = \frac{1}{m} dZ_1 \cdot X^T$$
$$db_1 = \frac{1}{m} \sum dZ_1$$



### 3. Parameter Updates
We use **Gradient Descent** to update parameters:
$$W := W - \alpha \cdot dW$$
$$b := b - \alpha \cdot db$$
*(Where $\alpha$ is the learning rate.)*

---

## 💻 Code Logic

### Data Preprocessing
* **Normalization:** We divide pixel values by 255. This scales inputs to a range of $[0, 1]$, preventing gradients from becoming too large and helping the model converge.
* **One-Hot Encoding:** The labels ($0-9$) are converted into vectors. For example, a label of `3` becomes $[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$.

### Core Functions
| Function | Purpose |
| :--- | :--- |
| `ReLU(Z)` | Activation function that replaces negative values with zero, allowing the network to learn non-linear patterns. |
| `softmax(Z)` | Converts raw output scores into probabilities that sum to 1. |
| `backward_prop()` | Uses the chain rule to determine how much each weight contributed to the total error. |
| `update_params()` | Subtracts a fraction of the gradient from the current weights to reduce error. |

---

## 🚀 Usage

1. **Prepare Data:** Ensure `train.csv` is in the project directory.
2. **Train:** Run the `gradient_descent` function.
   ```python
   W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
3. Visualize: Use test_prediction(index, W1, b1, W2, b2) to see the model's prediction on a visual plot of the digit.