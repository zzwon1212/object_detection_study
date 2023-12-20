#!/usr/bin/env python3

import time

import numpy as np
from function.convolution import Conv
import torch

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
  # torch_conv.weight = torch.nn.Parameter(torch.tensor(W))
  torch_conv.weight = torch.nn.Parameter(torch.from_numpy(W))
  for i in range(10):
    L3 = torch_conv(torch.tensor(X, requires_grad=False, dtype=torch.float32))
  # print(f"L3: {L3}, Shape: {L3.shape}")
  print(f"L3 time: {time.time() - l3_time}")

if __name__ == "__main__":
  convolution()
