import torch

data = [[1,2,4], [4,7,0]]

my_tensor = torch.tensor(data)
print(my_tensor)
print(my_tensor.shape)


shape = (3, 6)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

rand_tensor = torch.randn(shape)

print(f"ones tensor:\n {ones_tensor}")
print(f"zeros tensor: \n {zeros_tensor}")
print(f"random tensor: \n {rand_tensor}")


# create new tensor using sample shape and size of another
template_tensor = torch.tensor([[1,2], [3,7]])

replicate_tensor = torch.rand_like(template_tensor, dtype=torch.float)

print(replicate_tensor)

# Every tensor has three attributes, constantly used for debugging
print(f"Shape: {rand_tensor.shape}")
print(f"Datatype: {rand_tensor.dtype}")
print(f"Device: {zeros_tensor.device}")

# Shape: torch.Size([3, 6])
# Datatype: torch.float32  # why float type? => Because of Gradients (Tiny Adjustments) (weights, biases)
# Device: cpu

# AUTOGRAD  => Automatic Differentiation  (Build in Gradient Calculator)
# To enable  => requires_grad=True   # This tells pytorch engine that the current tensor is a PARAMETER. From now track * every single operation * that happens to it


# data tensor
x_data = torch.tensor([[4.5, 3.4], [0.2, 0.9]])

x_parameter = torch.tensor([[1.0], [0.5]], requires_grad=True)

print(f"Data tensor grad status: {x_data.requires_grad}")
print(f"Parameter tensor grad status: {x_parameter.requires_grad}")

#Data tensor grad status: False
#Parameter tensor grad status: True  # For requires_grad=True PyTorch begins to build a COMPUTATION GRAPH

# Lets build a Graph using example
# Goal: z = x * y, where y = a + b
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

x = torch.tensor(4.0, requires_grad=True)

y = a + b # First Operation
z = x * y # Second Operation

print(f"Result: {z}")  # Result: 20.0

# Now verify the COMPUTATION GRAPH the pytorch has built
# To verify, we can leverage .grad_fn, this points to the function that that created

# z was created by multiplication fun
print(f" grad_fn for z: {z.grad_fn}")   # grad_fn for z: <MulBackward0 object at 0x10ca1dbd0>

# y was created by addition fun
print(f" grad_fn for y: {y.grad_fn}")    # grad_fn for y: <AddBackward0 object at 0x10ca1dbd0>

# a was created by user, not an operation
print(f"grad_fn for a: {a.grad_fn}")   # grad_fn for a: None










