import os
import glob
import argparse
'''
Search out pdb files with multiple mods
'''
def main():
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-path', dest='path', type=str, help='Specify the directory to search', required=True)
    parser.add_argument('-output', dest='-output', type=str, help='Specify the output for the pdb files with multiple mods', required=True)
    args = parser.parse_args()

    path = args.path
    output = args.output

    fw = open(output, 'w')
    path = path
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
    print("done generating more models!")
if __name__ == "__main__":
    main() 
    

