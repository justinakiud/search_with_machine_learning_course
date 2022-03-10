import os
import random
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from nltk.stem import SnowballStemmer
import re
import pandas as pd
from nltk.tokenize import word_tokenize

directory = r'/workspace/search_with_machine_learning_course/data/pruned_products'
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing the products")
general.add_argument("--output", default="/workspace/datasets/fasttext/titles.txt", help="the file to output to")

# Consuming all of the product data takes a while. But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=0.1, type=float, help="The rate at which to sample input (default is 0.1)")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input

sample_rate = args.sample_rate

stemmer = SnowballStemmer("english")

def transform_training_data(name):
    tokens = word_tokenize(name)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]
    transformed_name = " ".join(tokens)
    return transformed_name

# Directory for product data

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                if (child.find('name') is not None and child.find('name').text is not None):
                    name = transform_training_data(child.find('name').text)
                    output.write(name + "\n")

NUM_KEEP = [10, 20, 50]
for id in NUM_KEEP:
    output_df = pd.read_table(output_file)
    output_df_split = output_df.iloc[:, 0].str.split(" ", 1, expand=True)
    output_df_split.rename(columns={0:'label', 1:'words'}, inplace=True)

    output_grouped = output_df_split.groupby('label').count().reset_index()

    keep_ids = output_grouped[output_grouped.words >= id]['label']
    keep_df = output_df_split[output_df_split['label'].isin(keep_ids)]

    output_new = keep_df["label"] + " " + keep_df["words"]
    output_new.to_csv(os.path.split(output_file)[0] + os.path.sep + "output_mincount_{}.txt".format(id), index=False, header=None)