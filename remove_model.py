import argparse
import os
def remove_pdb(input,mpath, output):
    with open(input,'r')as f:
        for line in f:
            row = line.strip().split(':')
            pdb = row[0]
            fw = open(output+ str(pdb) + '.pdb','w')
            with open(mpath+ str(pdb) + '.pdb','r')as fp:
                li = fp.readlines()
            for count in range(len(li)):
                if li[count].strip().split()[0] == 'MODEL' and li[count].strip().split()[1] == '2':
                    break
                fw.write(li[count].strip().split('\n')[0] + '\n')
                
                        
def main():
    parser = argparse.ArgumentParser(description='supply the old and new directory to update the downloaded pdbs for a specific ion i.e. ZN, CA, CO3')
    parser.add_argument('-input', dest='input', type=str, help='more_model_total file', required=True)
    parser.add_argument('-mpath', dest='mpath', type=str, help='Specify the location of pdb list i.e. data_list.txt file for the ion of interest', required=True)
    parser.add_argument('-output', dest='output', type=str, help='Specify the location for the output', required=True)
    
    args = parser.parse_args()
    
    
    input = args.input
    mpath = args.mpath
    output = args.output
    
    os.makedirs(output, exist_ok=True)
    
    remove_pdb(input, mpath, output)
    print("Done removing model!")
    

if __name__ == '__main__':
  main()
  
       

def remove_fa():
    fw = open('./data/fasta/fasta_one_model.fa','w')
    with open('./data/fasta/fasta.fa','r') as f:
        line = f.readlines()
    for count in range(len(line)):
        if line[count].strip().split()[0][0] == '>':
            pdb = line[count].strip().split()[0][1:5]
            chain = line[count].strip().split()[0][6]
            fw.write('>' + str(pdb) + '-' + str(chain) + '\n')
            ff = open('./data/fasta/single_fasta/' + str(pdb)+'-'+str(chain)+'.fa','w')
            ff.write('>' + str(pdb) + '-' + str(chain) + '\n')
            seq = line[count + 1].strip().split()[0]
            with open('./data/more_model_total.txt','r') as fp:
                for li in fp:
                    if pdb == li.strip().split(':')[0][:4] and chain == li.strip().split(':')[0][5]:
                            num = li.strip().split(':')[1]
                            seq_len = len(seq)/int(num)
                            seq = seq[:int(seq_len)]
                            # seq = ','.join(str(i) for i in seq)
                            break
            fw.write(str(seq) + '\n')
            ff.write(str(seq) + '\n')
            ff.close()

def remove_label():
    fw = open('./data/T_label_indep_one_model.txt','w')
    with open('./data/T_label_indep.txt','r') as f:
        line = f.readlines()
    for count in range(len(line)):
        if line[count].strip().split()[0][0] == '>':
            pdb = line[count].strip().split()[0][1:5]
            chain = line[count].strip().split()[0][6]
            fw.write('>' + str(pdb) + '-' + str(chain) + '\n')
            seq = line[count + 1].strip().split()[0]
            with open('data/more_model_total.txt', 'r') as fp:
                for li in fp:
                    if pdb == li.strip().split(':')[0][:4] and chain == li.strip().split(':')[0][5]:
                            num = li.strip().split(':')[1]
                            seq_len = len(seq)/int(num)
                            seq = seq[:int(seq_len)]
                            # seq = ','.join(str(i) for i in seq)
                            break
            fw.write(str(seq) + '\n')

