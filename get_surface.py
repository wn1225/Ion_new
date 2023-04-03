"""
   Calculate the surface

   1. input
      ag_ab_chain.txt (Calculate the surface of symmetric structure)
      antigen_antibody_new.txt (Calculate the surface of all pdb)
      Antigen-only pdb

   2. output
      surface.txt
"""

import uuid
from typing import Optional, Dict
from Bio.PDB import PDBParser, SASA


class SurfaceArea:
    def __init__(self, probe_radius: float = 1.4, n_points: int = 100, radii_dict: Optional[Dict] = None):
        if radii_dict is None:
            # radii_dict = {'X': 2.0, 'LI': 1.82, 'NE': 1.54, 'SI': 2.1, 'CL': 1.75, 'AR': 1.88, 'CU': 1.4, 'ZN': 1.39,
            #               'GA': 1.87, 'AS': 1.85, 'SE': 1.9, 'BR': 1.85, 'KR': 2.02, 'AG': 1.72, 'CD': 1.58, 'IN': 1.93,
            #               'SN': 2.17, 'TE': 2.06, 'XE': 2.16, 'PT': 1.72, 'AU': 1.66, 'HG': 1.55, 'TL': 1.96, 'PB': 2.02,
            #               'U': 1.86, 'EU': 1.47}
            radii_dict = {'X': 2.0}

        self.parser = PDBParser(QUIET=1)
        self.structure_computer = SASA.ShrakeRupley(probe_radius=probe_radius, n_points=n_points, radii_dict=radii_dict)

    def __call__(self, name, loc, chain) -> float:
        struct = self.parser.get_structure(name, loc)
        # self.structure_computer.compute(struct, level="C")
        # return struct[0]['C'].sasa
        self.structure_computer.compute(struct, level="R")
        return struct[0][chain].child_list


def get_surface(antigen_sasa):
    surface_list = ''
    for antigen in antigen_sasa:
        if float(antigen.sasa) > 1.0:
            surface_list += '1'
        else:
            surface_list += '0'
    return surface_list


def get_sasa_surface(complex_name, chain):
    an_path = './data/one_pdb/'
    antigen_pdb = complex_name[:4] + '_' + chain + '.pdb'
    sasa_fn = SurfaceArea()
    antigen_sasa = sasa_fn(uuid.uuid4().hex, an_path + antigen_pdb, chain)
    surface_list = get_surface(antigen_sasa)
    return surface_list

if __name__ == '__main__':
    fw = open('./data/surface/40%_surface_indep.txt','w')
    with open('./data/40%_indep_data_list.txt','r') as fp:
        for line in fp:
            name = line.strip().split('_')[0]
            chain = line.strip().split('_')[1]
            with open('./data/surface/surface_' + str(name) + '-' + str(chain) + '.txt', 'w') as la:
                la.write('>' + str(name) + '-' + str(chain) + '\n')
                la.write(get_sasa_surface(name, chain) + '\n')

                fw.write('>' + str(name) + '-' + str(chain) + '\n')
                fw.write(get_sasa_surface(name, chain) + '\n')
                print(str(name) + 'over')