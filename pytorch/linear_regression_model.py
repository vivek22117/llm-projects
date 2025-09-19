import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x) -> nn.Linear:
        return self.linear_layer(x)


model = LinearRegressionModel(in_features=1, out_features=1)
print("Model Architecture")
print(model)
# Model Architecture
# LinearRegressionModel(
#   (linear_layer): Linear(in_features=1, out_features=1, bias=True)
# )

import torch.optim as optim

# Hyperparameters
learning_rate = 0.01

# Create an Adam Optimizer
# we pass model.parameters to tell it which tenors to manage
# Manual cal: W -= lr * W.grad   || b -= lr * b.grad
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Pre-built loss func form torch.nn
# Manual calculation: loss = torch.mean((y_hat - y_true)**3)
loss_fn = nn.MSELoss()  # Mean Squared Error Loss


# Our batch of data will have 10 data points
N = 10
D_in = 1
D_out = 1

# Create our true target labels y using the true W and b
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
x = torch.randn(N, D_in)
y_true = x @ true_W + true_b + torch.randn(N, D_out) * 0.1

epochs = 100

for epoch in range(epochs):
    ## Forward Pass ##
    y_hat = model(x)

    ## Calculate Loss ##
    loss = loss_fn(y_hat, y_true)

    ## Zero Gradients ##
    optimizer.zero_grad()

    ## Compute Gradients ##
    loss.backward()

    ## Update the parameters ##
    optimizer.step()


    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}")


# Gradient descent result
# Epoch 00: Loss=9.7060
# Epoch 10: Loss=8.4886
# Epoch 20: Loss=7.3667
# Epoch 30: Loss=6.3481
# Epoch 40: Loss=5.4344
# Epoch 50: Loss=4.6231
# Epoch 60: Loss=3.9089
# Epoch 70: Loss=3.2850
# Epoch 80: Loss=2.7438
# Epoch 90: Loss=2.2779

# 2nd Round of execution
# Epoch 00: Loss=4.1756
# Epoch 10: Loss=3.5322
# Epoch 20: Loss=2.9519
# Epoch 30: Loss=2.4390
# Epoch 40: Loss=1.9937
# Epoch 50: Loss=1.6130
# Epoch 60: Loss=1.2919
# Epoch 70: Loss=1.0245
# Epoch 80: Loss=0.8047
# Epoch 90: Loss=0.6260