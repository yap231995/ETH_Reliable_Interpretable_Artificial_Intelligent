import torch
import torch.nn as nn

# fix seed so that random initialization always performs the same
torch.manual_seed(1)


# create the model N as described in the
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

x = torch.rand((1, 10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # make sure we can compute the gradient w.r.t x
t = 1 # target class
epsReal = 0.4 #depending on your data this might be large or small

eps = epsReal - 1e-7 # small constant to offset floating-point erros

print("Original Class: ", N(x).argmax(dim=1).item())
assert(N(x).argmax(dim=1).item() == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) #loss with respect to target t.
loss.backward() ##Give us dloss/dx

# your code here
# in x.grad you have access to the gradient of loss w.r.t. x
print("gradient wrt x", x.grad)
eta = eps * torch.sign(x.grad)
xBar = x - eta


print("New Class: ", N(xBar).argmax(dim=1).item())
assert(N(xBar).argmax(dim=1).item() == 1)
assert( torch.norm((x-xBar), p=float('inf')) <= epsReal)