import csv
import glob
import os


def convert_single_csv_to_segmentation_csv(input_file, output_file):
    with open(input_file) as f_in:
        with open(output_file) as f_out:
            input_reader = csv.DictReader(f_in)
            for row in input_reader:
                text = row["Conversation"]
                pass


def create_topic_segmentation_dataset(input_folder, output_folder):
    input_files = glob.glob(os.path.join(input_folder, "*.csv")
    for file in input_files:
        pass
