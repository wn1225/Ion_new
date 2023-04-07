import requests
import os
import argparse
from Bio.PDB import Select, PDBIO, PDBList
from Bio.PDB.PDBParser import PDBParser

# Download the PDB files from the RCSB database website


def main():
  parser=argparse.ArgumentParser(description='supply the input and outpaths to download the pdb files for a specific ion i.e. ZN, CA, CO3')
  parser.add_argument('-input', dest='file_path', type=str, help='Specify the location of pdb list i.e. data_list.txt file for the ion of interest', required=True)
  parser.add_argument('-output', dest='output_path', type=str, help='Specify the location for the pdb files to be stored', required=True)
  
  args = parser.parse_args()
  

  file_path = args.file_path
  output_path = args.output_path
  
  pdbl = PDBList()
  os.makedirs(output_path, exist_ok=True)
  with open(file_path, 'r') as fp:
    lines = fp.read().split('\n')
    for line in lines:
      id = line[:4]
      try:
        pdbl.retrieve_pdb_file(id,pdir=output_path,file_format="pdb")

      except:
        print('could not download pdb for '+id)
      os.system("mv "+output_path+"pdb"+id.lower()+".ent  "+output_path+id.upper()+".pdb")
  print("Done downloading PDB files!")
      
if __name__ == '__main__':
  main()