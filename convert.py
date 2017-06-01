# -*- coding: utf-8 -*-
from sys import argv

script, file1, file2  = argv

with open(file1) as infile, open(file2, 'w') as outfile:
    for line in infile.readlines():
        list = line.split(':', 1)
        outfile.write(list[0] + '\t' + list[1])