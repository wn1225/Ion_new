# Remove the 'Name of non-standard residue' line from the downloaded pdb file, starting each line with 'HETATM'
#输入pdb的路径，输出pdb的路径
def remove_HETATM(opath, npath):
    with open(opath, 'r') as op:
        write_all = ''
        for line in op:
            if line[:6] != 'HETATM':
                write_all += line
    with open(npath, 'w') as np:
        np.write(write_all)


if __name__ == '__main__':
    with open('./data/data_list.txt', 'r') as li:
        for line in li:
            name = line[:4].upper()
            # print(name)
            remove_HETATM('./data/pdb/' + str(name) + '.pdb', './data/pdb_update/' + str(name) + '.pdb')



    # remove_HETATM('./ploymer/complex_pdb/5ACO.pdb','./ploymer/complex_pdb_update/5ACO.pdb')


