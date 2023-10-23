#!/usr/bin/python3

# Usage: python3 test_to_input.py <test_set_path> <column> <output_path>

"""
This script is used to create the input file from the test_set.tsv file
The output is input.txt, and keeps only the column passed as parameter
All the lines where that column is empty are removed
"""

import pandas as pd
import csv, os, sys

HALFTRUE_AS_NONE = False # If True, the half-true examples will be considered as none. Otherwise, they will be considered as false

trues = ["true", "mostly true"]
if HALFTRUE_AS_NONE:
    falses = ["false"]
    nones = ["partly true/misleading", "complicated/hard to categorise", "half-true"]
else:
    falses = ["false", "partly true/misleading", "half-true"]
    nones = ["complicated/hard to categorise"]

# Read the parameters
if len(sys.argv) != 4:
    print("Usage: python3 test_to_input.py <test_set_path> <column> <output_path>")
    sys.exit(1)

test_set_path = sys.argv[1]
column = sys.argv[2]

# Load the dataset
dataset = pd.read_csv(test_set_path, sep="\t", header=0, quoting=csv.QUOTE_NONE, dtype={'label': str})

# Check that the column exists
if column not in dataset.columns:
    print("Error: the column " + column + " does not exist in the dataset")
    sys.exit(1)

if column != "claim" and column != "news-like" and column != "social-like":
    print("\033[93m" + "Warning: the column " + column + " is not supposed to be used as input" + "\033[0m")

# Remove the lines where the specified column is empty
dataset = dataset[dataset["news-like"].notnull()] # We want to keep only the lines where the news-like column is not empty, since this means the claim has no ambiguity

# Check the distribution of the labels, percentage of trues, falses and nones
trues_n = len(dataset[dataset["label"].str.lower().isin(trues)])
falses_n = len(dataset[dataset["label"].str.lower().isin(falses)])
nones_n = len(dataset[dataset["label"].str.lower().isin(nones)])
total_n = trues_n + falses_n + nones_n
if total_n != len(dataset):
    print("Error: the number of lines does not match the number of labels")
    print("Total number of lines: " + str(len(dataset)))
    print("Total number of labels: " + str(total_n))
    sys.exit(1)
print("Number of true: " + str(trues_n) + " (" + str(round(trues_n/total_n*100, 2)) + "%)")
print("Number of false: " + str(falses_n) + " (" + str(round(falses_n/total_n*100, 2)) + "%)")
print("Number of none: " + str(nones_n) + " (" + str(round(nones_n/total_n*100, 2)) + "%)")

# Keep only the column
dataset = dataset[[column]]

output_path = sys.argv[3]

# Save the input file. NOTE: the input file is named INPUT because it is the input of the model
input_path = os.path.join(output_path, "input.txt")
dataset.to_csv(input_path, index=False, header=False, sep="\t")

# Print the number of lines
print("Number of lines: " + str(len(dataset)))
print("Input file saved at " + input_path)
