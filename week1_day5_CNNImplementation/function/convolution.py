import numpy as np

class Conv:
  def __init__(self, batch, in_c, out_c, in_h, in_w, k_h, k_w, dilation, stride, padding):
    self.batch = batch
    self.in_c = in_c
    self.out_c = out_c
    self.in_h = in_h
    self.in_w = in_w
    self.k_h = k_h
    self.k_w = k_w
    self.dilation = dilation
    self.stride = stride
    self.padding = padding

    self.out_h = (in_h + 2*padding - k_h) // stride + 1
    self.out_w = (in_w + 2*padding - k_w) // stride + 1

  def check_range(self, a, b):
    return a > -1 and a < b

  # naive convolution. sliding window metric
  def conv(self, A, W):
    B = np.zeros((self.batch, self.out_c, self.out_h, self.out_w))

    for b in range(self.batch):
      for oc in range(self.out_c):
        # each channel of output
        for oh in range(self.out_h):
          for ow in range(self.out_w):
            # each pixel of output shape
            a_j = oh * self.stride - self.padding
            for kh in range(self.k_h):
              if not self.check_range(a_j, self.in_h):
                B[b, oc, oh, ow] += 0
              else:
                a_i = ow * self.stride - self.padding
                for kw in range(self.k_w):
                  if not self.check_range(a_i, self.in_w):
                    B[b, oc, oh, ow] += 0 # TODO 그냥 pass 하면 안 돼?
                  else:
                    B[b, oc, oh, ow] += np.dot(A[b, :, a_j, a_i], W[oc, :, kh, kw])
                  a_i += self.stride
              a_j += self.stride
    return B

  # IM2COL. change n-dim input to 2-dim matrix
  def im2col(self, A):
    mat = np.zeros((self.in_c * self.k_h * self.k_w, self.out_h * self.out_w), dtype=np.float32)

    mat_i = 0
    mat_j = 0
    for c in range(self.in_c):
      for kh in range(self.k_h):
        for kw in range(self.k_w):
          in_j = kh * self.dilation - self.padding
          for oh in range(self.out_h):
            if not self.check_range(in_j, self.in_h):
              for ow in range(self.out_w):
                mat[mat_j, mat_i] = 0
                mat_i += 1
            else:
              in_i = kw * self.dilation - self.padding
              for ow in range(self.out_w):
                if not self.check_range(in_i, self.in_w):
                  mat[mat_j, mat_i] = 0
                  mat_i += 1
                else:
                  mat[mat_j, mat_i] = A[0, c, in_j, in_i]
                  mat_i += 1
                in_i += self.stride
            in_j += self.stride
          mat_i = 0
          mat_j += 1
    return mat

  # GEMM. 2d matrix multiplication
  def gemm(self, A, W):
    a_mat = self.im2col(A)
    w_mat = W.reshape(W.shape[0], -1)
    b_mat = np.matmul(w_mat, a_mat)
    b_mat = b_mat.reshape([self.batch, self.out_c, self.out_h, self.out_w])
    return b_mat
