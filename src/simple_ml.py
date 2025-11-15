import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except Exception:  # pragma: no cover - optional C++ extension
    simple_ml_ext = None


def add(x, y):
    """Simple helper that mirrors numpy-style broadcasting semantics."""
    return x + y


def _read_gzip_array(filename):
    with gzip.open(filename, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic == 2051:  # images
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_items, rows * cols), magic
        elif magic == 2049:  # labels
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data, magic
        else:  # pragma: no cover
            raise ValueError(f"Unexpected MNIST magic {magic}")


def parse_mnist(image_filename, label_filename):
    images, _ = _read_gzip_array(image_filename)
    labels, _ = _read_gzip_array(label_filename)
    images = images.astype(np.float32) / 255.0
    return images, labels


def softmax_loss(Z: np.ndarray, y: np.ndarray):
    z_max = Z.max(axis=1, keepdims=True)
    stabilized = Z - z_max
    logsumexp = np.log(np.exp(stabilized).sum(axis=1))
    gold = stabilized[np.arange(Z.shape[0]), y]
    return np.mean(logsumexp - gold)


def _one_hot(indices, num_classes):
    eye = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    eye[np.arange(indices.shape[0]), indices] = 1.0
    return eye


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    num_classes = theta.shape[1]
    for start in range(0, X.shape[0], batch):
        end = start + batch
        xb = X[start:end]
        yb = y[start:end]
        logits = xb @ theta
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        targets = _one_hot(yb, num_classes)
        grad = xb.T @ (probs - targets) / xb.shape[0]
        theta -= lr * grad


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    num_classes = W2.shape[1]
    for start in range(0, X.shape[0], batch):
        end = start + batch
        xb = X[start:end]
        yb = y[start:end]
        hidden = xb @ W1
        relu_mask = hidden > 0
        hidden = np.maximum(hidden, 0)

        logits = hidden @ W2
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        targets = _one_hot(yb, num_classes)

        grad_logits = (probs - targets) / xb.shape[0]
        grad_W2 = hidden.T @ grad_logits

        grad_hidden = grad_logits @ W2.T
        grad_hidden *= relu_mask
        grad_W1 = xb.T @ grad_hidden

        W2 -= lr * grad_W2
        W1 -= lr * grad_W1



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
