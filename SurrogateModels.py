# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:23:24 2020

@author: scorrado3

SurrogateModels object containing all relevant surrogate models
"""

import os
import pickle

outputs = {'out_1': 'LR',
           'out_2': 'LR'}

class SurrogateModels(object):
   
    model_folder = 'models'
    outputs = outputs
    
    def __init__(self):
        for key in self.outputs:
            path = os.path.join(self.model_folder, self.outputs[key], key+'.pickle')
            setattr(self, key, pickle.load(open(path, 'rb')))
        
    

