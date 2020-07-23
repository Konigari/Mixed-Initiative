import os
from os import path
import shutil

already_existing_list = ['./ASSIGNMENTS CSV/' + x for x in os.listdir('./ASSIGNMENTS CSV')]
already_existing_txts = [x[:4]+'.txt' for x in os.listdir('./ASSIGNMENTS CSV')]

rachna_list = []
vijay_list = []
saurabh_list = []

dataset = '../dataset/train/'
files  = os.listdir(dataset)
required_files = []
for file in files:
	if file not in already_existing_txts:
		required_files.append('../dataset/train/'+file)

dataset = '../dataset/dev/'
files = os.listdir(dataset)

for file in files:
	if file not in already_existing_txts:
		required_files.append(dataset+file)
size = len(required_files)
for index in range(size):
	if index <= 533:
		shutil.copy(required_files[index], './vijay')
	elif index >= 534 and index <= 805:
		shutil.copy(required_files[index], './rachna')
	else:
		shutil.copy(required_files[index], './saurabh')


