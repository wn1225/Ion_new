import requests
import os
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser

# Download the PDB files from the RCSB database website

def download_pdb(file_path,output_path):
  fp = open(file_path, 'r')
  for line in fp:
    list = line[:4]
    try:
      r = requests.get('https://files.rcsb.org/download/' + str(list) + '.pdb', stream=True)
      print(format(list), "already downloaded", )
      os.makedirs(output_path, exist_ok=True)
      with open('./data/pdb/' + str(list) + '.pdb', 'wb') as f:
        f.write(r.content)
    except:
      print(''+list+' did not work')



if __name__ == '__main__':

  file_path = r'./data/data_list.txt'
  output_path = r'./data/pdb'
  download_pdb(file_path,output_path)
