# from Archi_1 import blackbox as bb_1
# from Archi_2 import blackbox as bb_2
import subprocess
import os
import sys
best_acc_train_1=[]
best_acc_train_2=[]
best_acc_valid_1=[]
best_acc_valid_2=[]


for i in range(20):
    os.system('python ./blackbox.py')
    #os.system('python ./Archi_2/blackbox.py')



