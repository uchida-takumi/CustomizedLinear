#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np

from CustomizedLinear import CustomizedLinear

# setting
dtype = torch.float

# size of layers
Dim_INPUT  = 2
Dim_HIDDEN = 5
Dim_OUTPUT = 1

# mask matrix of INPUT-HIDDEN whose elements are 0 or 1.
get_bin_matrix = lambda Dim0, Dim1 : np.random.choice([0,1], size=(Dim0, Dim1))
mask  = torch.tensor(get_bin_matrix(Dim_INPUT, Dim_HIDDEN), dtype=dtype)

# create randomly input x
batch = 1
x = torch.randn(batch, Dim_INPUT, dtype=dtype)

# create randomly output y 
y = torch.randn(batch, Dim_OUTPUT, dtype=dtype)

# pipe as model
model = torch.nn.Sequential(
        CustomizedLinear(mask, bias=None),
        torch.nn.Linear(Dim_HIDDEN, Dim_OUTPUT, bias=None),
        )

# backward pass
print('=== mask matrix ===')
print(mask)
print('===================')
learning_rate = 0.1
for t in range(3):
    # forward
    y_pred = model(x)

    # loss
    loss = (y_pred - y).abs().mean()

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Use autograd to compute the backward pass
    loss.backward()

    # Update the weights
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
            # check masked param.grad
            if np.array(param.grad).size == np.array(mask).size:
                print('epoch={}, loss={}'.format(t,loss.item()))
                print('------')
                print('↓↓↓masked weight↓↓↓')
                print(param.t())
                print('↓↓↓masked grad of weight↓↓↓')
                print(param.grad.t())
    
