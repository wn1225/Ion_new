import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-input', dest='input', type=str, help='Specify the location of file that contain the pdb chains i.e. data_list.txt for the specific ion of interest', required=True)
    parser.add_argument('-ion-path', dest='ipath', type=str, help='Specify the directory for the ion under consideration', required=True)
    parser.add_argument('-residues', dest='residues', type=str, help='Specify the candidate residues', required=True)

    args = parser.parse_args()

    input = args.input
    ipath = args.ipath
    residues = args.residues
    
    r = residues.replace(',','')
    ion = ipath.split('/')[1]
    residue_list = list(r)
    
    fw = open(ipath+'positive.txt', 'w')
    with open(input,'r')as fp:
        for lines in fp:
            row = lines.strip().split()
            pdb = row[0][:4]
            chain = row[0][5]
            # fw.write('>'+pdb.upper() + '-' + chain + '\n')

            with open(ipath+'pdb_update/{}.pdb'.format(pdb), 'r') as f:
                for line1 in f:
                    row = line1.strip().split('\n')
                    if (row[0][:4] == 'LINK'):
                        row[0] = row[0].replace(" ","")
                        row[0] = row[0][4:]
                        if row[0][:2] == ion:
                            row[0] = row[0][5:]
                            n = 0
                            for i in range(6): # 把数字全部删除
                                if row[0][int(i)].isdigit() == True:
                                    n += 1
                            row[0] = row[0][n:]
                            a = ''
                            n = 0
                            for i in range(len(row[0])):
                                if (row[0][i].isdigit() == False) or (row[0][i].isdigit() == True and i<5):
                                    a += row[0][i]
                                    n += 1
                                else:
                                    if row[0][i].isdigit() == True: break
                            row[0] = row[0][n:]
                            c = a[-1]
                            radius = a[-4:-1]
                            if c == chain:
                                # radius = row[15:18]
                                if radius == 'GLY':
                                    radius = 'G'
                                elif radius == 'ALA':
                                    radius = 'A'
                                elif radius == 'VAL':
                                    radius = 'V'
                                elif radius == 'LEU':
                                    radius = 'L'
                                elif radius == 'ILE':
                                    radius = 'I'
                                elif radius == 'PRO':
                                    radius = 'P'
                                elif radius == 'PHE':
                                    radius = 'F'
                                elif radius == 'TYR':
                                    radius = 'Y'
                                elif radius == 'TRP':
                                    radius = 'W'
                                elif radius == 'SER':
                                    radius = 'S'
                                elif radius == 'THR':
                                    radius = 'T'
                                elif radius == 'CYS':
                                    radius = 'C'
                                elif radius == 'MET':
                                    radius = 'M'
                                elif radius == 'ASN':
                                    radius = 'N'
                                elif radius == 'GLN':
                                    radius = 'Q'
                                elif radius == 'ASP':
                                    radius = 'D'
                                elif radius == 'GLU':
                                    radius = 'E'
                                elif radius == 'LYS':
                                    radius = 'K'
                                elif radius == 'ARG':
                                    radius = 'R'
                                elif radius == 'HIS':
                                    radius = 'H'
                                if len(radius) == 1 and radius in residue_list:
                                    index = row[0][:-12]
                                    fw.write(str(pdb) + '_' + str(chain) + ' ' + str(radius) + ' ' + str(chain) + ' ' + str(
                                        index) + '\n')
                        else:
                         if (ion in row[0]) == True:
                            a = ''
                            n = 0
                            for i in range(len(row[0])):
                                if (row[0][i].isdigit() == False) or (row[0][i].isdigit() == True and i < 4):
                                    a += row[0][i]
                                    n += 1
                                else:
                                    if row[0][i].isdigit() == True:break
                            row[0] = row[0][n:]
                            c = a[-1]
                            radius = a[-4:-1]
                            index = ''
                            n = 0
                            flag = 0
                            for i in range(len(row[0])):
                                if row[0][int(i)].isdigit() == True:
                                    index += row[0][int(i)]
                                    n += 1
                                else:
                                    if row[0][int(i)] == ion[0]:
                                        flag = 1
                                        continue
                                    if row[0][int(i)] == ion[1] and flag == 1:   # index 1A ZN ZN
                                        break
                                    index += row[0][int(i)]
                                    n += 1

                            row[0] = row[0][n:]
                            # if ('ZN' in row[0]) == True :
                            if c == chain:
                                # radius = row[15:18]
                                if radius == 'GLY':
                                    radius = 'G'
                                elif radius == 'ALA':
                                    radius = 'A'
                                elif radius == 'VAL':
                                    radius = 'V'
                                elif radius == 'LEU':
                                    radius = 'L'
                                elif radius == 'ILE':
                                    radius = 'I'
                                elif radius == 'PRO':
                                    radius = 'P'
                                elif radius == 'PHE':
                                    radius = 'F'
                                elif radius == 'TYR':
                                    radius = 'Y'
                                elif radius == 'TRP':
                                    radius = 'W'
                                elif radius == 'SER':
                                    radius = 'S'
                                elif radius == 'THR':
                                    radius = 'T'
                                elif radius == 'CYS':
                                    radius = 'C'
                                elif radius == 'MET':
                                    radius = 'M'
                                elif radius == 'ASN':
                                    radius = 'N'
                                elif radius == 'GLN':
                                    radius = 'Q'
                                elif radius == 'ASP':
                                    radius = 'D'
                                elif radius == 'GLU':
                                    radius = 'E'
                                elif radius == 'LYS':
                                    radius = 'K'
                                elif radius == 'ARG':
                                    radius = 'R'
                                elif radius == 'HIS':
                                    radius = 'H'
                                if len(radius) == 1 and radius in residue_list:
                                    # index = row[9:-12]
                                    fw.write(str(pdb) + '_' + str(chain) + ' ' + str(radius) + ' ' + str(chain) + ' ' + str(
                                        index) + '\n')
    print("done generating positive examples!")

if __name__ == '__main__':

    main()

