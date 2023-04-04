import argparse
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser

class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0


def main():    
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-input', dest='input', type=str, help='Specify the location of file that contain the pdb chains i.e. data_list.txt for the specific ion of interest', required=True)
    parser.add_argument('-output', dest='output', type=str, help='Specify the location of pdb list i.e. data_list.txt file for the ion of interest', required=True)

    args = parser.parse_args()

    input = args.input
    output = args.output
    
    with open(input,'r') as fp:
        lines = fp.read().split('\n')
        for line in lines:
            line = line.strip().split()
            name = line[0][:4].upper()
            chain = line[0][5]
            p = PDBParser(PERMISSIVE=1)
            pdb_file = output+'{}.pdb'.format(name)
            structure = p.get_structure(pdb_file, pdb_file)

            # for chain in chains:
            pdb_chain_file = output+'{}_{}.pdb'.format(name,chain)
            io_w_no_h = PDBIO()
            io_w_no_h.set_structure(structure)
            io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chain))
            print(pdb_chain_file)
    print("Done running divide_pdb.py")
if __name__ == '__main__':
    main()