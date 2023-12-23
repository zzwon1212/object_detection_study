#!/usr/bin/env python3

import time
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from function.convolution import Conv
from function.pooling import Pool
from function.fc import FC
from function.activation import *

np.random.seed(12)

def convolution():
  # define the shape of input & weight
  in_w = 6 # 3
  in_h = 6 # 3
  in_c = 3 # 1
  out_c = 16
  batch = 1
  k_h = 3
  k_w = 3

  # X = np.arange(9, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
  X = np.arange(in_w * in_h * in_c * batch, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
  W = np.array(np.random.standard_normal([out_c, in_c, k_h, k_w]), dtype=np.float32)

  # print(X)
  print(f"Shape of X: {X.shape}")
  print(f"Shape of W: {W.shape}")

  Convolution = Conv(batch=batch,
                     in_c=in_c,
                     out_c=out_c,
                     in_h=in_h,
                     in_w=in_w,
                     k_h=k_h,
                     k_w=k_w,
                     dilation=1,
                     stride=1,
                     padding=0)

  # 1. naive conv
  l1_time = time.time()
  for i in range(10):
    L1 = Convolution.conv(X, W)
  # print(f"L1: {L1}, Shape: {L1.shape}")
  print(f"L1 time: {time.time() - l1_time}")

  # 2. IM2COL & GEMM conv
  l2_time = time.time()
  for i in range(10):
    L2 = Convolution.gemm(X, W)
  # print(f"L2: {L2}, Shape: {L2.shape}")
  print(f"L2 time: {time.time() - l2_time}")

  # 3. PyTorch conv
  l3_time = time.time()
  torch_conv = torch.nn.Conv2d(in_c,
                               out_c,
                               kernel_size=k_h,
                               stride=1,
                               padding=0,
                               bias=False,
                               dtype=torch.float32)
  torch_conv.weight = torch.nn.Parameter(torch.tensor(W))
  for i in range(10):
    L3 = torch_conv(torch.tensor(X, requires_grad=False, dtype=torch.float32))
  # print(f"L3: {L3}, Shape: {L3.shape}")
  print(f"L3 time: {time.time() - l3_time}")

def forward_net():
  batch = 1
  in_c = 3
  in_w = 6
  in_h = 6
  k_h = 3
  k_w = 3
  out_c = 1

  X = np.arange(batch * in_c * in_w * in_h, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
  W1 = np.array(np.random.standard_normal([out_c, in_c, k_h, k_w]), dtype=np.float32)

  Convolution = Conv(batch=batch,
                     in_c=in_c,
                     out_c=out_c,
                     in_h=in_h,
                     in_w=in_w,
                     k_h=k_h,
                     k_w=k_w,
                     dilation=1,
                     stride=1,
                     padding=0)

  L1 = Convolution.gemm(X, W1)
  print(f"L1:\n{L1}, Shape: {L1.shape}")

  Pooling = Pool(batch=batch,
                 in_c=1,
                 out_c=1,
                 in_h=4,
                 in_w=4,
                 kernel=2,
                 dilation=1,
                 stride=2,
                 padding=0)

  L1_MAX = Pooling.pool(L1)
  print(f"L1_MAX:\n{L1_MAX}, Shape: {L1_MAX.shape}")

  W2 = np.array(np.random.standard_normal([1, L1_MAX.shape[1] * L1_MAX.shape[2] * L1_MAX.shape[3]]), dtype=np.float32)
  Fc = FC(batch=L1_MAX.shape[0])
  L2 = Fc.fc(L1_MAX, W2)
  print(f"L2:\n{L2}, Shape: {L2.shape}")

def plot_activation():
  x = np.arange(-10, 10, 1)

  out_relu = relu(x)
  out_leaky = leaky_relu(x)
  out_sigmoid = sigmoid(x)
  out_tanh = tanh(x)

  plt.plot(x, out_relu, 'r', label='relu')
  plt.plot(x, out_leaky, 'b', label='leaky')
  plt.plot(x, out_sigmoid, 'g', label='sigmoid')
  plt.plot(x, out_tanh, 'bs', label='tanh')
  plt.ylim([-2, 2])
  plt.legend()
  plt.savefig("activation_graph.png")

def shallow_network():
  # input: [1, 1, 6, 6], iter: 2
  X = [np.random.standard_normal([1, 1, 6, 6]),
       np.random.standard_normal([1, 1, 6, 6])]
  Y = np.array([1, 1], dtype=np.float32)

  # Conv weights, [1, 1, 3, 3], [1, 4]
  W1 = np.random.standard_normal([1, 1, 3, 3])
  W2 = np.random.standard_normal([1, 4])

  padding = 0
  stride = 1

  # Shape of L1 layer
  L1_h = (X[0].shape[2] + 2*padding - W1.shape[2]) // stride + 1
  L1_w = (X[0].shape[3] + 2*padding - W1.shape[3]) // stride + 1

  # Conv
  Convolution = Conv(batch=X[0].shape[0],
                     in_c=X[0].shape[1],
                     out_c=W1.shape[0],
                     in_h=X[0].shape[2],
                     in_w=X[0].shape[3],
                     k_h=W1.shape[2],
                     k_w=W1.shape[3],
                     dilation=1,
                     stride=stride,
                     padding=padding)

  # Conv diff
  Conv_diff = Conv(batch=X[0].shape[0],
                   in_c=X[0].shape[1],
                   out_c=W1.shape[0],
                   in_h=X[0].shape[2],
                   in_w=X[0].shape[3],
                   k_h=L1_h,
                   k_w=L1_w,
                   dilation=1,
                   stride=1,
                   padding=0)

  # Max Pooling
  Pooling = Pool(batch=X[0].shape[0],
                 in_c=W1.shape[0],
                 out_c=W1.shape[0],
                 in_h=L1_h,
                 in_w=L1_w,
                 kernel=2,
                 dilation=1,
                 stride=2,
                 padding=0)

  # FC
  Fc = FC(batch=X[0].shape[0])

  num_epoch = 100
  for e in range(num_epoch):
    total_loss = 0
    for i in range(len(X)):
      # Forward
      L1 = Convolution.gemm(X[i], W1)
      L1_a = sigmoid(L1)
      L1_MAX = Pooling.pool(L1_a)

      L2 = Fc.fc(L1_MAX, W2)
      L2_a = sigmoid(L2)

      loss = np.square(Y[i] - L2_a) * 0.5

      total_loss += loss

      # Backward
      # 1. dE / dW2
      diff_E__L2_a = L2_a - Y[i]

      diff_L2_a__L2 = L2_a * (1 - L2_a)

      diff_L2__W2 = L1_MAX.reshape([1, -1])

      diff_E__W2 = diff_E__L2_a * diff_L2_a__L2 * diff_L2__W2

      # 2. dE / dW1
      diff_E__L2_a = L2_a - Y[i]

      diff_L2_a__L2 = L2_a * (1 - L2_a)

      diff_L2__L1_MAX = W2.reshape([1, 1, 2, 2])
      diff_L2__L1_MAX = diff_L2__L1_MAX.repeat(2, axis=2).repeat(2, axis=3)

      L1_MAX_repeat = L1_MAX.repeat(2, axis=2).repeat(2, axis=3)
      diff_L1_MAX__L1_a = np.equal(L1_MAX_repeat, L1_a).astype(int)

      diff_L1_a__L1 = L1_a * (1 - L1_a)

      diff_L1__w1 = X[i]

      diff_E__W1 = diff_E__L2_a * diff_L2_a__L2 * diff_L2__L1_MAX * diff_L1_MAX__L1_a * diff_L1_a__L1
      diff_E__W1 = Conv_diff.gemm(diff_L1__w1, diff_E__W1)
      # print("diff_E__W1\n", diff_E__W1)

      # Update
      W2 = W2 - 0.01 * diff_E__W2
      W1 = W1 - 0.01 * diff_E__W1

    print(f"{e} epoch loss {total_loss / len(X)}")

if __name__ == "__main__":
  # convolution()
  # forward_net()
  # plot_activation()
  shallow_network()
