import torch

# 1. Initialize weight with gradient tracking enabled
w = torch.tensor(2.0, requires_grad=True)

# Hyperparameters & Data
x = 3.0
target = 10.0
learning_rate = 0.05

# 2. Forward pass: compute prediction and MSE loss
prediction = w * x                      # 2.0 * 3.0 = 6.0
loss = (prediction - target) ** 2       # (6.0 - 10.0)^2 = 16.0

# 3. Backward pass: compute d(loss)/dw via autograd engine
loss.backward()

print(f"Initial Loss : {loss.item():.4f}")
print(f"Gradient (w) : {w.grad.item():.4f}")

# 4. Optimization step (Gradient Descent): theta = theta - lr * grad
# We use torch.no_grad() so PyTorch doesn't track the update step itself
with torch.no_grad():
    w -= learning_rate * w.grad

print(f"Updated w    : {w.item():.4f}")

# Verify that loss actually decreased:
new_loss = (w * x - target) ** 2
print(f"New Loss     : {new_loss.item():.4f}")