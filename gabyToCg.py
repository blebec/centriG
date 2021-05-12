#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:11:03 2021

@author: cdesbois
"""

import os

from bs4 import BeautifulSoup

import config

paths = config.build_paths()



dirname = os.path.join(paths['owncFig'], 'data', 'gabyToCg', 'sources')

filename = os.path.join(dirname, '4100gg3_vm_cp_cx_max.svg')
# filename = os.path.expanduser('~/4100gg3_vm_cp_cx_max.svg')



#%%
with open(filename, 'r') as f:
    data = f.read()
    
Bs_data = BeautifulSoup(data, 'xml')

# b_unique = Bs_data.find_all()
# print(b_unique)


import xml.etree.ElementTree as ET

tree = ET.parse(filename)

roots = tree.getroot()
print(roots)

for root in roots:
    print(list(root))
# -> a metadata

# nb properties in [tag, attributes, text string, tail string, child element]

labels = set()

gs = []
lines = []
polylines = []

root = roots[1]
for item in list(root):
    # check if no child
    if not list(item):
        print('no elements for {}'.format(item.tag))
        continue
    # list child
    for ob in list(item):
        print('='*20)
        print(ob)
        label = ob.tag.split('}')[-1]
        labels.add(label)
        if label == 'g':
            gs.append(ob.attrib)
        elif label == 'line':
            lines.append(ob.attrib)
        elif label == 'polyline':
            polylines.append(ob.attrib)
        else:
            print('{} is not a referenced label'.format(label))
        print('label = {}'.format(label))
        print('contains for {}'.format(ob.tag))
      #  print('attrib= {}'.format(ob.attrib))
        print('txt= {}'.format(ob.text))
        print('list= {}'.format(list(ob)))
        print('-'*20)        
        
#%%

roots = tree.getroot()
        
#%%
import xmltodict

xmltodict.parse(filename)
