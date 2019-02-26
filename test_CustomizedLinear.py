#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is test code for CustomizedLinear.py
"""

import unittest
from CustomizedLinear import CustomizedLinear

import numpy as np
import torch

class test_CustomizedLinear(unittest.TestCase):
    
    def setUp(self):
        print('== setUp ==')
        
    def test_case01(self):
        """
        Can it solve bellow problem? 
        ----------------------------       
        0*x_0 + 1*x_1 + 2*x_2 + 3*x_3 = y_0
        9*x_0 + 0*x_1 + 0*x_2 + 6*x_3 = y_1
        2*x_0 + 0*x_1 + 4*x_2 + 0*x_3 = y_2
        """
        answer_weight = [
                [0, 9, 2],
                [1, 0, 0],
                [2, 0, 4],
                [3, 6, 0],
                ]
        mask = (np.array(answer_weight)>0).astype(int)
        CL = CustomizedLinear(mask, bias=False)
        
        x = np.random.rand(1000, 4)
        y = np.dot(x, answer_weight)
        train(CL, x, y)
        predicted_weight = list(CL.parameters())[0].t()
        
        error = np.array(answer_weight) - predicted_weight.data.numpy()
        self.assertLess(error.mean(), 0.01)
        
    def test_case02(self):
        """
        Can it work as one of multiple layer?
        """
        mask0 = np.array([
                    [0,1,1,1],
                    [1,1,0,1],
                ])
        mask1 = np.array([
                    [0,1,1,1],
                    [1,1,0,1],
                    [1,0,0,1],
                    [0,1,1,1],
                ])
        mask2 = np.array([
                    [0,1],
                    [1,1],
                    [1,0],
                    [1,1],
                ])
        
        def get_sequantial():
            CL0 = CustomizedLinear(mask0)
            CL1 = CustomizedLinear(mask1)
            CL2 = CustomizedLinear(mask2)
            
            sequencial = torch.nn.Sequential(
                    CL0, torch.nn.ReLU(), CL1, torch.nn.ReLU(), CL2)
            
            return sequencial
        
        answer_sequencial = get_sequantial()
        train_sequencial  = get_sequantial()
        
        x = torch.tensor(np.random.rand(1000, 2), dtype=torch.float32)
        y = answer_sequencial(x)
        
        train_x, train_y = x[:800], y[:800]
        test_x,  test_y  = x[800:], y[800:]
        
        train(train_sequencial, train_x, train_y, epoch=10)
        predict_y = train_sequencial(test_x)
        abs_error = abs(predict_y - test_y)
        abs_error_rate = abs_error.sum() / abs(test_y).sum()
        
        # Assertion
        self.assertLess(abs_error_rate.item(), 0.05)
        
    def tearDown(self):
        print('== tearDown ==')


def train(model, x, y, epoch=10):
    _x = torch.tensor(x, dtype=torch.float32)
    _y = torch.tensor(y, dtype=torch.float32)
    criterion = torch.nn.L1Loss(reduction='mean')    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for t in range(epoch):
        for i in range(_x.shape[0]):
            __x = _x[[i]]
            __y = _y[[i]]
            forward = model(__x)
            loss = criterion(forward, __y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
if __name__ == '__main__':
    unittest.main()


    