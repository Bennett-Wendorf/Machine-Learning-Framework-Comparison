import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from data: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy: \n {x_np} \n")

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3, 4)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")

print("GPU available: ", torch.cuda.is_available(), "\n")

tensor = torch.ones(4, 4)

# This computes the matrix multiplication between two tensors. Y1, y2, y3 will have the same value
# ``Tensor.T`` is used to transpose the tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(f"y1: \n {y1} \n")
print(f"y2: \n {y2} \n")
print(f"y3: \n {y3} \n")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f"z1: \n {z1} \n")
print(f"z2: \n {z2} \n")
print(f"z3: \n {z3} \n")

# In-place operations are denoted by a ``_`` suffix. For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.
print(f"tensor: \n {tensor} \n")
tensor.add_(5)
print(f"tensor: \n {tensor} \n")

# The use of in-place operations is discouraged because history is immediately lost, so computing derivatives can be problematic.

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: \n {t} \n")
n = t.numpy()
print(f"n: \n {n} \n")

t.add_(1)
print(f"t: \n {t} \n")
print(f"n: \n {n} \n")