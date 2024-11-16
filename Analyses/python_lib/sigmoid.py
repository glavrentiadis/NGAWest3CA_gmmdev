#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 14:58:46 2023

@author: glavrent
"""
import numpy as np

def sigmoid(x):
    '''
    Sigmoid function
    '''
    
    return 1 / (1 + np.exp(-x))
