#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from CustomizedLinear import CustomizedLinear

# define mask matrix to customize linear
mask = torch.tensor(
  [[1, 0, 1],
   [0, 1, 0],
   [1, 0, 1],
   [1, 0, 1],]
  )

# define size of layers.
# this architect is [INPUT, HIDDEN(masked(customized) linear), OUTPUT]-layers.
Dim_INPUT  = mask.shape[0]
Dim_HIDDEN = mask.shape[1]
Dim_OUTPUT = 1

# create randomly input:x, output:y as train dataset.
batch = 1
x = torch.randn(batch, Dim_INPUT)
y = torch.randn(batch, Dim_OUTPUT)

# pipe as model
model = torch.nn.Sequential(
        CustomizedLinear(mask, bias=None), # dimmentions is set from mask.size 
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
            # mask is also saved in param, but mask.requires_grad=False
            if param.requires_grad: 
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