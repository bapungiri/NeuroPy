#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:02:04 2019

@author: bapung
"""

import sys

def DataDirPath():
    
    comp = sys.platform
    
    if comp == 'linux':
        
        DataPath = '/data/DataGen/'
        
    else:
        
        DataPath = '../../DataGen/'
        
    return DataPath