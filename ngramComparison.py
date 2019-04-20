# import csv

# with open('./swda(1)employee_birthda y.txt') as csv_file:
# csv_reader = csv.reader(csv_file, delimiter=',')
# line_count = 0
# for row in csv_reader:
#     if line_count == 0:
#         print(f'Column names are {", ".join(row)}')
#         line_count += 1
#     else:
#         print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
#         line_count += 1
# print(f'Processed {line_count} lines.')
import os
import argparse
import pathlib
import csv
import itertools
from nltk.util import ngrams
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize     
# from nltk.tokenize
stop_words = set(stopwords.words('english')) 


sentences = []

def get_all_files_in_dir(location): 
    for subdir, dirs, files in os.walk(location):
        for file in files:
            yield os.path.join(subdir, file)

def is_CSV(filepath):
    return filepath.endswith(".txt") and not "metadata" in filepath

def get_all_csvs_in_dir(location):
    return filter(is_CSV, get_all_files_in_dir(location))
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())    
    c = a.intersection(b)
    return c
def stop_word_removal(str):
    word_tokens = word_tokenize(str)
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence
name = './Switchboard-Corpus/swda_data/train'
for file in os.listdir(name):

    f_output = open("./output/"+file+"_output.txt",'w')
    # pathlib.Path(os.path.join(command.output_location, file_without_ext)).mkdir(parents=True, exist_ok=True)
    with open(name+"/"+file,'r') as f:
        print(f.name)
        for sentence in f.readlines():
            # print(sentence.split("|")[1])
            sentences.append(sentence.split("|")[1])
        # sentences = np.array(saved_column.tolist())
        # # print(len(saved_column))
        temp = []

        for sentence in sentences:
            s = stop_word_removal(sentence)
            temp.append(" ".join(s))
        # for x in itertools.combinations(temp, 2):
        for i in range(1,len(temp)):
            for j in range(2,len(temp)):
                words = word_tokenize(temp[i])
                wordsFiltered = []
                 
                for w in words:
                    if w not in stop_words:
                        wordsFiltered.append(w)
                similarity = get_jaccard_sim(temp[i],temp[j])
                depth = j- i
                if (len(similarity) > 3 and depth > 2):
                    f_output.write("similarity"+"==>" + " ".join(similarity)+" | " + "depth : "+ str(depth) + "\n")

        
        # print("---------------done -------------------")
        # for i in ennumeratesentences:
        #     sentences
        # # rows = csv.DictReader(f)
        # sentences = [row for row in rows]
        # print(sentences)


