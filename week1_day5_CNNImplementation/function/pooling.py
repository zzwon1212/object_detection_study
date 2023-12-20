import numpy as np

# 2D pooling
class Pool:
  def __init__(self, batch, in_c, out_c, in_h, in_w, kernel, dilation, stride, padding):
    self.batch = batch
    self.in_c = in_c
    self.out_c = out_c
    self.in_h = in_h
    self.in_w = in_w
    self.kernel = kernel
    self.dilation = dilation
    self.stride = stride
    self.padding = padding
    self.out_w = (in_w + padding*2 - kernel) // stride + 1
    self.out_h = (in_h + padding*2 - kernel) // stride + 1

  def pool(self, A):
    B = np.zeros([self.batch, self.out_c, self.out_h, self.out_w], dtype=np.float32)
    for b in range(self.batch):
      for c in range(self.in_c):
        for oh in range(self.out_h):
          a_j = oh * self.stride - self.padding
          for ow in range(self.out_w):
            a_i = ow * self.stride - self.padding
            B[b, c, oh, ow] = np.amax(A[b, c, a_j : a_j+self.kernel, a_i : a_i+self.kernel])
    return B
