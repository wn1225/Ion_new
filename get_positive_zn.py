fw = open('./data/positive_CDEH_new_new.txt', 'w')
with open('./data/data_list.txt','r')as fp:
    for lines in fp:
        row = lines.strip().split()
        pdb = row[0][:4]
        chain = row[0][5]
        # fw.write('>'+pdb.upper() + '-' + chain + '\n')

        with open('./data/pdb_update/{}.pdb'.format(pdb), 'r') as f:
            for line1 in f:
                row = line1.strip().split('\n')
                if (row[0][:4] == 'LINK'):
                    row[0] = row[0].replace(" ","")
                    row[0] = row[0][4:]
                    if row[0][:2] == 'ZN':
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
                            if len(radius) == 1 and radius in ['C','H','E','D']:
                                index = row[0][:-12]
                                fw.write(str(pdb) + '_' + str(chain) + ' ' + str(radius) + ' ' + str(chain) + ' ' + str(
                                    index) + '\n')
                    else:
                      if ('ZN' in row[0]) == True:
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
                                if row[0][int(i)] == 'Z':
                                    flag = 1
                                    continue
                                if row[0][int(i)] == 'N' and flag == 1:   # index 1A ZN ZN
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
                            if len(radius) == 1 and radius in ['C','H','E','D']:
                                # index = row[9:-12]
                                fw.write(str(pdb) + '_' + str(chain) + ' ' + str(radius) + ' ' + str(chain) + ' ' + str(
                                    index) + '\n')



