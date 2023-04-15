import argparse
import glob
import os
import zipfile

import os
flist = []
def getFlist(path,npath,file):
    num = 0
    for file_abs in glob.glob(path):
        num += 1
        print(file_abs)
    with open(npath,'r')as fp:
        lines = fp.readlines()
    for i in range(1,num+1):
        new_name = lines[(i-1)*2].strip().split()[0][1:]
        os.rename(path[:-1] + file + '_' + str(i) + '.pssm', npath[:-11] +new_name+'.pssm')


    return flist

def un_zip(file_name):
   """unzip zip file"""
   zip_file = zipfile.ZipFile(file_name)
   if os.path.isdir(file_name + "_files"):
       pass
   else:
       os.mkdir(file_name + "_files")
   for names in zip_file.namelist():
       zip_file.extract(names,file_name + "_files/")
   zip_file.close()


DIR = "data/"
ion = 'CA/'
# un_zip(DIR+ion+'CA_pssm_files.zip')
e = DIR+ion+'CA_pssm_files.zip_files/pssm/afb83db55ffc3cd8844651bc1fbfcab6/*'
print(e)
getFlist(e,'data/CA/no_pssm.txt','afb83db55ffc3cd8844651bc1fbfcab6')
