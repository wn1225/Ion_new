import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-input', dest='input', type=str, help='Specify the location of file that contain the pdb chains i.e. data_list.txt for the specific ion of interest', required=True)
    parser.add_argument('-model-one-path', dest='model_one', type=str, help='Specify the path to model-one-pdb', required=True)
    parser.add_argument('-fasta-dir', dest='fasta', type=str, help='Specify the location for the fasta directory', required=True)
    parser.add_argument('-one-model-output', dest='one_model_output', type=str, help='Specify the one_model_output directory', required=True)
    
    
    args = parser.parse_args()

    input = args.input
    model_one = args.model_one
    fasta = args.fasta
    # one_model_output = args.one_model_output
    os.makedirs(fasta, exist_ok=True)
    # os.makedirs(one_model_output, exist_ok=True)

    fw = open(fasta + 'fasta_total.fa', 'w')
    # fw = open('./data/ind264_data/ind264_data_list.txt','w')

    with open(input,'r')as fp:
        for lines in fp:
            row = lines.strip().split()
            pdb = row[0][:4].upper()
            chain = row[0][5]
            fw.write('>'+pdb.upper() + '-' + chain + '\n')

            ff = open(fasta + str(pdb) + '-' + str(chain) + '.fa','w')
            ff.write('>'+pdb.upper() + '-' + chain + '\n')

            with open(model_one+'{}_{}.pdb'.format(pdb,chain), 'r') as f:
                m = -10000
                for line1 in f:
                    row = line1.strip().split()
                    if (row[0] == 'ATOM' and row[4][0] == chain):
                        if (len(row[2]) == 7):
                            row[7] = row[6]
                            row[6] = row[5]
                            row[5] = row[4]
                            row[4] = row[3]
                            row[3] = row[2][4:]
                            row[2] = row[2][:4]
                        if (len(row[2]) == 8):
                            row[8] = row[7]
                            row[7] = row[6]
                            row[6] = row[5]
                            row[5] = row[4]
                            row[4] = row[3]
                            row[3] = row[2][4:]
                            row[2] = row[2][:4]
                        if (len(row[4]) != 1):
                            row.append('0')
                            # row[11] = row[10]
                            row[7] = row[6]
                            row[6] = row[5]
                            row[5] = row[4][1:]
                            row[4] = row[4][0]
                        if (len(row[3]) != 3):
                            row[3] = row[3][-3:]

                        if str(m) != row[5]:  # 氨基酸改变
                            # print(row[5])
                            # print(m)
                            # print(row[3])
                            if row[3] == 'GLY':
                                fw.write('G')
                                ff.write('G')
                            elif row[3] == 'ALA':
                                fw.write('A')
                                ff.write('A')
                            elif row[3] == 'VAL':
                                fw.write('V')
                                ff.write('V')
                            elif row[3] == 'LEU':
                                fw.write('L')
                                ff.write('L')
                            elif row[3] == 'ILE':
                                fw.write('I')
                                ff.write('I')
                            elif row[3] == 'PRO':
                                fw.write('P')
                                ff.write('P')
                            elif row[3] == 'PHE':
                                fw.write('F')
                                ff.write('F')
                            elif row[3] == 'TYR':
                                fw.write('Y')
                                ff.write('Y')
                            elif row[3] == 'TRP':
                                fw.write('W')
                                ff.write('W')
                            elif row[3] == 'SER':
                                fw.write('S')
                                ff.write('S')
                            elif row[3] == 'THR':
                                fw.write('T')
                                ff.write('T')
                            elif row[3] == 'CYS':
                                fw.write('C')
                                ff.write('C')
                            elif row[3] == 'MET':
                                fw.write('M')
                                ff.write('M')
                            elif row[3] == 'ASN':
                                fw.write('N')
                                ff.write('N')
                            elif row[3] == 'GLN':
                                fw.write('Q')
                                ff.write('Q')
                            elif row[3] == 'ASP':
                                fw.write('D')
                                ff.write('D')
                            elif row[3] == 'GLU':
                                fw.write('E')
                                ff.write('E')
                            elif row[3] == 'LYS':
                                fw.write('K')
                                ff.write('K')
                            elif row[3] == 'ARG':
                                fw.write('R')
                                ff.write('R')
                            elif row[3] == 'HIS':
                                fw.write('H')
                                ff.write('H')
                            m = str(row[5])
            fw.write('\n')
            ff.write('\n')

    print("Done extracting fasta!")   
        
if __name__ == "__main__":
    main() 