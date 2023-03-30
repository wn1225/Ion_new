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


if __name__ == '__main__':
    a = ''
    chains = list(a)
    with open('./data/data_list.txt','r') as fp:
        for line in fp:
            line = line.strip().split()
            name = line[0][:4].upper()
            chain = line[0][5]
    # chains = ['A','G','H','I','J','K','L']
            p = PDBParser(PERMISSIVE=1)
            pdb_file = './data/pdb_update/{}.pdb'.format(name)
            structure = p.get_structure(pdb_file, pdb_file)

            # for chain in chains:
            pdb_chain_file = './data/model_one_pdb/{}_{}.pdb'.format(name,chain)
            io_w_no_h = PDBIO()
            io_w_no_h.set_structure(structure)
            io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chain))
            print(pdb_chain_file)