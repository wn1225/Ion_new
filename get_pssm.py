import os
import re
import codecs  # 或者io，使用哪种包无所谓
import pandas as pandas

def PSSM(pdb_name,chain):
    # with open('./pdb/fasta/' + pdb_name + '_fasta.txt','')
    os.system(
        r'psiblast -query "D:\LenovoSoftstore\Install\pycharm2022\pythonProject\IonBindingSturcturalAnalysis\data\fasta\single_fasta\\'
        + str(pdb_name) + '-' + str(chain)
        + '.fa"'
        + r' -db D:\LenovoSoftstore\Install\ncbi_blast\blast-BLAST_VERSION+\db\swissprot -evalue 0.001 -num_iterations 3'
        + r' -out_ascii_pssm "D:\LenovoSoftstore\Install\pycharm2022\pythonProject\IonBindingSturcturalAnalysis\data\pssm\1\\' +
        str(pdb_name) + '-' + str(chain) +'.pssm"')


    print(str(pdb_name) + " PSSM over")

# PSSM('6WPS')
if __name__ == '__main__':
    with open('./data/data_list.txt','r') as fp:
        for line in fp:
            row = line.strip().split('_')
            pdb_name = row[0]
            chain = row[1]
            PSSM(pdb_name,chain)