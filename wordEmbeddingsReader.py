#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 05:58:34 2017

@author: diana
"""

import numpy as np

class GloVeFileReader:
    def __init__(self, filePath):
        self.filePath = filePath
        
    def readWordEmbeddings(self):
        f = open(self.filePath, 'r')
        wordEmbeddings = {}
        
        for line in f.readlines():
            tokens = line.split()
            wordEmbeddings[tokens[0]] = np.array(tokens[1:])            
        
        f.close()
        
        return wordEmbeddings