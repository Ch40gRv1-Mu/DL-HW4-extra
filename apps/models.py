import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                device=device,
                dtype=dtype,
            ),
            nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x: ndl.Tensor):
        return self.block(x)


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        blocks = [
            ConvBatchNorm(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBatchNorm(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
                    ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
                )
            ),
            ConvBatchNorm(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBatchNorm(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
                    ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
                )
            ),
        ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.head(self.features(x))


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn', 'lstm', or 'transformer'
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_len = seq_len
        self.seq_model = seq_model
        builders = {
            "rnn": lambda: (nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype), hidden_size),
            "lstm": lambda: (nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype), hidden_size),
            "transformer": lambda: (
                nn.Transformer(
                    embedding_size,
                    hidden_size,
                    num_layers,
                    device=device,
                    dtype=dtype,
                    batch_first=False,
                    sequence_len=seq_len,
                ),
                embedding_size,
            ),
        }
        if seq_model not in builders:
            raise ValueError(f"Unsupported sequence model: {seq_model}")
        self.model, projection_in = builders[seq_model]()
        self.projection = nn.Linear(projection_in, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape

        x = self.embedding(x)
        if self.seq_model == "transformer":
            x, h = self.model(x)
        else:
            x, h = self.model(x, h)
        x = x.reshape((seq_len * bs, -1))
        return self.projection(x), h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)
