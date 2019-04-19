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

# from nltk.tokenize

argp = argparse.ArgumentParser()
argp.add_argument("input_location")
argp.add_argument("output_location")
command = argp.parse_args()
saved_column = []

def get_all_files_in_dir(location): 
    for subdir, dirs, files in os.walk(location):
        for file in files:
            yield os.path.join(subdir, file)

def is_CSV(filepath):
    return filepath.endswith(".csv") and not "metadata" in filepath

def get_all_csvs_in_dir(location):
    return filter(is_CSV, get_all_files_in_dir(location))
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())    
    c = a.intersection(b)
    return c

for file in itertools.islice(get_all_csvs_in_dir(command.input_location),0,1,1):
    file_without_ext = os.path.splitext(file)[0]
    pathlib.Path(os.path.join(command.output_location, file_without_ext)).mkdir(parents=True, exist_ok=True)
    with open(file) as f:
        df = pd.read_csv(f)
        saved_column = df['text'] #you can also use df['column_name']
        print(type(saved_column))
        
        saved_column = np.array(saved_column.tolist())
        print(len(saved_column))
        for i in range(1,len(saved_column)):
            for j in range(2,len(saved_column)):
                similarity = get_jaccard_sim(saved_column[i],saved_column[j])
                depth = j- i
                if (len(similarity) > 3 and depth > 2):
                    print("similarity",similarity,"depth",depth)
        print("---------------done -------------------")
        # for i in ennumeratesaved_column:
        #     saved_column
        # # rows = csv.DictReader(f)
        # sentences = [row for row in rows]
        # print(sentences)


