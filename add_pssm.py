
num = 0
with open('./data/label/40%_T_label.txt', 'r') as la:
    with open('./data/feature/40%_new_feature.txt', 'w') as nfe:
        with open('./data/feature/40%_feature_no_pssm.txt', 'r') as fe:
            num_acid = fe.readline().strip()
            nfe.write(num_acid + '\n')
            for i in range(int(num_acid)):
                name_pssm = la.readline().strip().split()[0]
                la.readline()
                name_lines = name_pssm[1:].split('_')
                # name = name_lines[0] + '-' + name_lines[1]
                name = name_lines[0]

                chain = name_lines[1]

                with open('./data/pssm/' + str(name) + '-' + str(chain) + '.pssm', 'r') as ps:
                    ps.readline()
                    ps.readline()
                    ps.readline()
                    num_atom = fe.readline().strip()
                    nfe.write(num_atom + '\n')
                    line = ps.readline().strip().split()
                    for i_a in range(int(num_atom.split()[0])):
                        # print(line[0])
                        try:
                             seq_ac_pssm = int(line[0]) - 1
                        except:
                            print(name)
                        feature_a = fe.readline().strip()
                        nfe.write(feature_a + '\t')
                        seq_ac = feature_a.split()[0]
                        if seq_ac_pssm != int(seq_ac):
                            line = ps.readline().strip().split()
                            seq_ac_pssm += 1
                        pssm = line[2:22]
                        for num_p in range(pssm.__len__()):
                            nfe.write(pssm[num_p] + '\t')
                        nfe.write('\n')


# with open('new_data_feature_surface.txt', 'r') as n1:
#     with open('new_data_feature_surface(1).txt', 'r') as n2:
#         i = 1
#         while True:
#             line1 = n1.readline().rstrip()
#             line2 = n2.readline().rstrip()
#             if line1 == '' and line2 == '':
#                 print('SUCCESS!!!')
#                 break
#             if line2 == line1:
#                 line1 = n1.readline().rstrip()
#                 line2 = n2.readline().rstrip()
#                 print(i)
#                 i += 1
#             else:
#                 print('ERROR!!!!!')
