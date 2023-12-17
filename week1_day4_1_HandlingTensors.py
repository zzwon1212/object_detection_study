#!/usr/bin/env python3

import torch
import numpy as np

def make_tensor():
  # int16
  a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
  # float
  b = torch.tensor([2], dtype=torch.float32)
  # double
  c = torch.tensor([3], dtype=torch.float64)

  print(f"{a}\n{b}\n{c}\n")

  print(f"shape of tensor {a.shape}")
  print(f"data type of tensor {a.dtype}")
  print(f"device tensor is stored on {a.device}")

def sumsub_tensor():
  a = torch.tensor([3, 2])
  b = torch.tensor([5, 3])
  print(f"input {a}, {b}\n")

  sum = a + b
  print(f"sum: {sum}\n")

  sub = a - b
  print(f"sub: {sub}\n")

  sum_elements_a = a.sum()
  print(f"sum_elements: {sum_elements_a}")

def mul_tensor():
  a = torch.arange(0, 9).view(3, 3)
  b = torch.arange(0, 9).view(3, 3)
  print(f"input\n{a},\n{b}\n")

  # mat mul
  c = torch.matmul(a, b)
  print(f"mat mul\n{c}\n")

  # elementwise mul
  d = torch.mul(a, b)
  print(f"elementwise mul\n{d}")

def reshape_tensor():
  a = torch.tensor([2, 4, 5, 6, 7, 8])
  print(f"input\n{a}\n")

  # view
  b = a.view(2, 3)
  print(f"view\n{b}\n")

  # transpose
  b_t = b.t()
  print(f"transpose\n{b_t}")

def access_tensor():
  a = torch.arange(1, 13).view(4, 3)
  print(f"input\n{a}\n")

  # slicing
  print("slicing")
  print(a[:, 0]) # first row
  print(a[0, :]) # first col
  print(a[1, 1]) # [1, 1]

def transform_numpy():
  a = torch.arange(1, 13).view(4, 3)
  print(f"input\n{a}\n")

  a_np = a.numpy()
  print(f"tensor to numpy\n{a_np}\n")

  b = np.array([1, 2, 3])
  b_torch = torch.from_numpy(b)
  print(f"numpy to tensor\n{b_torch}")

def concat_tensor():
  a = torch.arange(1, 10).view(3, 3)
  b = torch.arange(1, 10).view(3, 3)
  c = torch.arange(1, 10).view(3, 3)
  print(f"input\n{a}\n{b}\n{c}\n")

  abc_0 = torch.concat([a, b, c], dim=0)
  print(f"concat (dim=0)\n{abc_0}, {abc_0.shape}\n")

  abc_1 = torch.concat([a, b, c], dim=1)
  print(f"concat (dim=1)\n{abc_1}, {abc_1.shape}")

def stack_tensor():
  a = torch.arange(1, 10).view(3, 3)
  b = torch.arange(1, 10).view(3, 3)
  c = torch.arange(1, 10).view(3, 3)
  print(f"input\n{a}\n{b}\n{c}\n")

  abc_0 = torch.stack([a, b, c], dim=0)
  print(f"stack (dim=0)\n{abc_0}, {abc_0.shape}\n")

  abc_1 = torch.stack([a, b, c], dim=1)
  print(f"stack (dim=1)\n{abc_1}, {abc_1.shape}")

def transpose_tensor():
  a = torch.arange(1, 10).view(3, 3)
  a_t = torch.transpose(a, 0, 1)
  print(f"input\n{a}")
  print(f"transpose\n{a_t}\n")

  b = torch.arange(1, 25).view(4, 3, 2)
  b_t = torch.transpose(b, 0, 2)
  print(f"input\n{b}, {b.shape}")
  print(f"transpose\n{b_t}\n{b_t.shape}\n")

  b_permute = b.permute(2, 0, 1)
  print(f"permute\n{b_permute}, {b_permute.shape}")

def quiz1():
  A = torch.arange(1, 7).view(2, 3)
  B = torch.arange(1, 7).view(2, 3)
  print(f"input\n{A}\n{B}")

  sum_AB = A + B
  print(f"Sum A and B\n{sum_AB}")

  sub_AB = A - B
  print(f"Subtract A and B\n{sub_AB}")

  sum_elements_A = A.sum()
  sum_elements_B = B.sum()
  print(f"Sum all elements of each A and B\n{sum_elements_A}\n{sum_elements_B}")

def quiz2():
  A = torch.arange(1, 46).view(1, 5, 3, 3)
  print(f"input\n{A}\n{A.shape}")

  A_t = torch.transpose(A, 1, 3)
  print(f"transpose\n{A_t}\n{A_t.shape}")

  print(A_t[0, 2, 2, :])

def quiz3():
  A = torch.arange(1, 7).view(2, 3)
  B = torch.arange(1, 7).view(2, 3)
  print(f"input\n{A}\n{B}")

  concat_AB = torch.concat([A, B], 1)
  print(f"concat\n{concat_AB}")

  stack_AB = torch.stack([A, B], 1)
  print(f"stack\n{stack_AB}")

if __name__ == "__main__":
  # make_tensor()
  # sumsub_tensor()
  # mul_tensor()
  # reshape_tensor()
  # access_tensor()
  # transform_numpy()
  # concat_tensor()
  # stack_tensor()
  # transpose_tensor()
  # quiz1()
  # quiz2()
  # quiz3()
