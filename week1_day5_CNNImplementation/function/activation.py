import numpy as np

def relu(x):
  x_shape = x.shape
  x = np.reshape(x, [-1])
  x = np.array([max(v, 0) for v in x], dtype=np.float32)
  x = np.reshape(x, x_shape)
  return x

def leaky_relu(x):
  x_shape = x.shape
  x = np.reshape(x, [-1])
  x = np.array([max(v, v*0.1) for v in x], dtype=np.float32)
  x = np.reshape(x, x_shape)
  return x

def sigmoid(x):
  x_shape = x.shape
  x = np.reshape(x, [-1])
  x = np.array([1 / (1 + np.exp(-v)) for v in x], dtype=np.float32)
  x = np.reshape(x, x_shape)
  return x

def tanh(x):
  x_shape = x.shape
  x = np.reshape(x, [-1])
  x = np.array([np.tanh(v) for v in x], dtype=np.float32)
  x = np.reshape(x, x_shape)
  return x
