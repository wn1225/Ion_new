import os
import glob
'''
Search out pdb files with multiple mods
'''

fw = open('./data/more_model_total1.txt', 'w')
path = '.\data\one_pdb\*'
for file_abs in glob.glob(path):
    # print(file_abs)
    # list.append(i)
    max = 0
    with open(file_abs,'r')as f:
        for line in f:
            row = line.strip().split()
            if len(row) == 2 and row[0] == 'MODEL':
                if max < int(row[1]):
                    max = int(row[1])
    if max != 0:
         fw.write(file_abs[-10:-4] + ':' + str(max) + '\n')

