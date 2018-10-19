#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:05:40 2018

Dataset Object

CHECK MAX DISBALANCE OPN REPLICATION FOR MULTICLASS

@author: ereyes
"""

import numpy as np

class dataset(object):
    
    """
    Constructor
    """
    def __init__(self, data_array, data_labels, BATCH_SIZE):
        self.BATCH_COUNTER = 0
        self.BATCH_COUNTER_EVAL = 0
        self.BATCH_SIZE = BATCH_SIZE
        self.data_array = data_array
        self.data_label = data_labels
    
    def get_batch(self):
        if(self.BATCH_COUNTER+self.BATCH_SIZE<self.data_array.shape[0]):
            batch_image = self.data_array[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            batch_label = self.data_label[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            self.BATCH_COUNTER += self.BATCH_SIZE
        else:
            self.BATCH_COUNTER = 0
            self.shuffle_data()
            batch_image = self.data_array[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            batch_label = self.data_label[self.BATCH_COUNTER:self.BATCH_COUNTER+self.BATCH_SIZE,...]
            self.BATCH_COUNTER += self.BATCH_SIZE
        
        return batch_image, batch_label
    
    def get_batch_eval(self):
        if(self.BATCH_COUNTER_EVAL+self.BATCH_SIZE<self.data_array.shape[0]):
            batch_image = self.data_array[self.BATCH_COUNTER_EVAL:self.BATCH_COUNTER_EVAL+self.BATCH_SIZE,...]
            batch_label = self.data_label[self.BATCH_COUNTER_EVAL:self.BATCH_COUNTER_EVAL+self.BATCH_SIZE,...]
            self.BATCH_COUNTER_EVAL += self.BATCH_SIZE
        else:
            left_samples = self.data_array.shape[0]-self.BATCH_COUNTER_EVAL
            batch_image = self.data_array[self.BATCH_COUNTER:self.BATCH_COUNTER+left_samples,...]
            batch_label = self.data_label[self.BATCH_COUNTER:self.BATCH_COUNTER+left_samples,...]
            self.BATCH_COUNTER_EVAL = 0
        
        return batch_image, batch_label
    
    def shuffle_data(self):
        idx = np.arange(self.data_array.shape[0])
        np.random.shuffle(idx)        
        self.data_array = self.data_array[idx,...]
        self.data_label = self.data_label[idx,...]
        


