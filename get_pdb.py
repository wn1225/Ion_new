import requests
import os
import argparse
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser

# Download the PDB files from the RCSB database website


def main():
  parser=argparse.ArgumentParser(description='supply the ion for which you want to download the pdb files i.e. ZN, CA, CO3')
  parser.add_argument('-input', dest='file_path', type=str, help='Specify the location of pdb list i.e. data_list.txt file for the ion of interest', required=True)
  parser.add_argument('-output', dest='output_path', type=str, help='Specify the location for the pdb files to be stored', required=True)
  
  args = parser.parse_args()
  
  
  # ion = args.ion 
  file_path = args.file_path
  output_path = args.output_path
  
  
  with open(file_path, 'r') as fp:
    lines = fp.read().split('\n')
    for line in lines:
      id = line[:4]
      try:
        r = requests.get('https://files.rcsb.org/download/' + str(id) + '.pdb', stream=True)
        print(format(id), "already downloaded", )
        os.makedirs(output_path, exist_ok=True)
        with open(output_path + str(id) + '.pdb', 'wb') as f:
          f.write(r.content)
      except:
        print('could not download pdb for '+id)
    
      
if __name__ == '__main__':
  main()