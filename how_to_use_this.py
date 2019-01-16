#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np

from CustomizedLinear import CustomizedLinear

# setting
dtype = torch.float

# size of layers
Dim_IN = 2
Dim_H = 5
Dim_OUT = 1

# mask matrix whose elements are 0 or 1.
get_bin_matrix = lambda Dim0, Dim1 : np.random.choice([0,1], size=(Dim0, Dim1))
mask  = torch.tensor(get_bin_matrix(Dim_IN, Dim_H), dtype=dtype)

# randomly create input x
batch = 2**12
x = torch.randn(batch, Dim_IN, dtype=dtype)

# create y on answer_w
y = torch.randn(batch, Dim_OUT, dtype=dtype)

# pipe as model
model = torch.nn.Sequential(
        CustomizedLinear(mask, bias=None),
        torch.nn.Linear(Dim_H, Dim_OUT, bias=None),
        )
# backward pass
learning_rate = 1e-2
for t in range(1000):
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
                if t % 100 == 0:
                    print('epoch={}, loss={}'.format(t,loss.item()))
                    print('------')
                    print('↓↓↓masked grad of weight↓↓↓')
                    print(param.grad.t())
    
