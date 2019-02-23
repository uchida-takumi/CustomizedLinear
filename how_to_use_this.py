#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np

from CustomizedLinear import CustomizedLinear

# size of layers
Dim_INPUT  = 2
Dim_HIDDEN = 5
Dim_OUTPUT = 1

# mask matrix of INPUT-HIDDEN whose elements are 0 or 1.
get_bin_matrix = lambda Dim0, Dim1 : np.random.choice([0,1], size=(Dim0, Dim1))
mask  = torch.tensor(get_bin_matrix(Dim_INPUT, Dim_HIDDEN))

# create randomly input x
batch = 1
x = torch.randn(batch, Dim_INPUT)

# create randomly output y 
y = torch.randn(batch, Dim_OUTPUT)

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
                print('--- epoch={}, loss={} ---'.format(t,loss.item()))
                print('↓↓↓masked weight↓↓↓')
                print(param.t())
                print('↓↓↓masked grad of weight↓↓↓')
                print(param.grad.t())
    

""" print result

    === mask matrix ===
    tensor([[1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0]])
    ===================
    --- epoch=0, loss=0.18818187713623047 ---
    ↓↓↓masked weight↓↓↓
    tensor([[-0.4230, -0.3346,  0.4397,  0.2637,  0.0868],
            [ 0.0000,  0.0000,  0.1355,  0.0000,  0.0000]]
            , requires_grad=True)
    ↓↓↓masked grad of weight↓↓↓
    tensor([[ 0.0238,  0.0391,  0.0316, -0.0255,  0.0274],
            [ 0.0000,  0.0000, -0.5167,  0.0000,  0.0000]])
    --- epoch=1, loss=0.1592715084552765 ---
    ↓↓↓masked weight↓↓↓
    tensor([[-0.4254, -0.3386,  0.4365,  0.2663,  0.0840],
            [ 0.0000,  0.0000,  0.1885,  0.0000,  0.0000]]
            , requires_grad=True)
    ↓↓↓masked grad of weight↓↓↓
    tensor([[ 0.0242,  0.0394,  0.0324, -0.0258,  0.0273],
            [ 0.0000,  0.0000, -0.5300,  0.0000,  0.0000]])
    --- epoch=2, loss=0.1262778639793396 ---
    ↓↓↓masked weight↓↓↓
    tensor([[-0.4278, -0.3425,  0.4331,  0.2689,  0.0813],
            [ 0.0000,  0.0000,  0.2441,  0.0000,  0.0000]]
            , requires_grad=True)
    ↓↓↓masked grad of weight↓↓↓
    tensor([[ 0.0246,  0.0397,  0.0339, -0.0260,  0.0272],
            [ 0.0000,  0.0000, -0.5554,  0.0000,  0.0000]])

    """