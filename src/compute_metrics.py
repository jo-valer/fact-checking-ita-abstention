#!/usr/bin/python3

# Usage: python3 compute_metrics.py <script_name> <test_set_file_path> <inference_file_path> <output_file_path>

"""
This script computes the metrics for the given model script, on a given test set.
It expects to find the inference file with the same number of lines as the test set (except header and empty lines).
For each line, it checks if the inference is correct or not, and computes accuracy, f1, and others.
NOTE: there are three possible labels: true, false, and half-true. We consider half-true as false.
"""

import pandas as pd
import os, sys, csv

HALFTRUE_AS_NONE = False # If True, the half-true examples will be considered as none. Otherwise, they will be considered as false

trues = ["true", "mostly true"]
if HALFTRUE_AS_NONE:
    falses = ["false"]
    nones = ["partly true/misleading", "complicated/hard to categorise", "half-true"]
else:
    falses = ["false", "partly true/misleading", "half-true"]
    nones = ["complicated/hard to categorise"]

# Read the parameters
if len(sys.argv) != 5:
    print("Usage: python3 compute_metrics.py <script_name> <test_file_path> <inference_file_path> <output_file_path>")
    sys.exit(1)

script_name = sys.argv[1]
test_set_file_path = sys.argv[2]
inference_file_path = sys.argv[3]
output_file_path = sys.argv[4]

# Load the test set
test_set = pd.read_csv(test_set_file_path, sep="\t", header=0, quoting=csv.QUOTE_NONE, dtype={'label': str})

# Get the name of the column used depending on the folder path: if the grandparent folder is "claim", then the column is "claim", otherwise it is "social-like"
parent_folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(output_file_path))))
if parent_folder == "claim":
    # column = "claim"
    column = "news-like" # We want to keep only the lines where the news-like column is not empty, since this means the claim has no ambiguity
elif parent_folder == "news-like":
    column = "news-like"
elif parent_folder == "social-like":
    column = "social-like"
else:
    print("Error: the grandparent folder of the output file must be either 'claim', 'news-like' or 'social-like'")
    sys.exit(1)

# Remove the lines where the column is empty
test_set = test_set[test_set[column].notnull()]

# Load the inference file
inference = pd.read_csv(inference_file_path, sep="\t", header=0, quoting=csv.QUOTE_NONE, dtype={'output_label': str})

# Check that the number of lines is the same
if len(test_set) != len(inference):
    print("Error: the number of lines in the test set is different from the number of lines in the inference file")
    print("Test set: " + str(len(test_set)) + " lines")
    print("Inference: " + str(len(inference)) + " lines")
    sys.exit(1)

queries_n = len(test_set)

# Lowercase the labels
test_set["label"] = test_set["label"].str.lower()
inference["output_label"] = inference["output_label"].str.lower()

# Compute the accuracy
correct = 0
correct_true = 0 # TRUE POSITIVE
total_true = len(test_set[test_set["label"] == "true"]) + len(test_set[test_set["label"] == "mostly true"])
correct_false = 0 # TRUE NEGATIVE
incorrect = 0
incorrect_true = 0 # FALSE NEGATIVE
incorrect_false = 0 # FALSE POSITIVE
if HALFTRUE_AS_NONE:
    total_false = len(test_set[test_set["label"] == "false"])
else:
    total_false = len(test_set[test_set["label"] == "false"]) + len(test_set[test_set["label"] == "partly true/misleading"]) + len(test_set[test_set["label"] == "half-true"]) # we consider half-true as false
none_dataset = inference[inference["output_label"] == "none"]
total_none = len(none_dataset)
# print("Total NONE: " + str(total_none))
correct_by_dataset = {}

for i in range(len(test_set)):
    predicted = inference["output_label"].iloc[i]
    expected = test_set["label"].iloc[i]
    if predicted == "half-true": predicted = "false"
    if expected in trues: expected = "true"
    elif expected in falses: expected = "false"
    elif expected in nones: expected = "none"
    if predicted == expected:
        correct += 1
        if expected == "true": correct_true += 1
        elif expected == "false": correct_false += 1
    else:
        incorrect += 1
        # Here the 'none' output is considered as incorrect in any case (as false positive or false negative, depending on the expected label)
        if expected == "true": incorrect_true += 1
        elif expected == "false": incorrect_false += 1

# Compute accuracies
accuracy = round(correct / len(test_set) * 100, 2)
accuracy_true = round(correct_true / total_true * 100, 2)
accuracy_false = round(correct_false / total_false * 100, 2)

# # Compute the F1 score
# precision = correct_true / (correct_true + incorrect_false)
# recall = correct_true / (correct_true + incorrect_true)
# f1 = round(2 * ((precision * recall) / (precision + recall)), 2)

# Compute macro-averaged F1 score
if correct_true == 0:
    f1_true = 0
else:
    precision_true = correct_true / (correct_true + incorrect_false)
    recall_true = correct_true / (correct_true + incorrect_true)
    f1_true = 2 * ((precision_true * recall_true) / (precision_true + recall_true))

if correct_false == 0:
    f1_false = 0
else:
    precision_false = correct_false / (correct_false + incorrect_true)
    recall_false = correct_false / (correct_false + incorrect_false)
    f1_false = 2 * ((precision_false * recall_false) / (precision_false + recall_false))

macro_f1 = round((f1_true + f1_false) / 2, 2)

# Percentage of NONE, over the number of incorrect predictions
wrong_predictions = len(test_set) - correct
if wrong_predictions == 0:
    percentage_none = 0
    percentage_pure_errors = 0
else:
    percentage_none = round((total_none / wrong_predictions) * 100, 2)
    percentage_pure_errors = round(((wrong_predictions - total_none) / wrong_predictions) * 100, 2)

# Abstention rate (i.e. NONE over the total number of predictions)
abstention_rate = round((total_none / len(test_set)) * 100, 2)

# Pure error rate (i.e. the number of pure errors over the total number of predictions)
pure_error_rate = round(((wrong_predictions - total_none) / len(test_set)) * 100, 2)

# Save the results
with open(output_file_path, "w") as f:
    f.write("EVALUATION RESULTS\n\n")
    f.write("Script name: " + script_name + "\n")
    f.write("Test set file: " + os.path.basename(test_set_file_path) + "\n")
    f.write("Number of queries: " + str(queries_n) + "\n")
    f.write("Accuracy: " + str(accuracy) + "%\n")
    f.write("Macro F1 score: " + str(macro_f1) + "\n")
    f.write("\n")
    f.write("Accuracy on TRUE: " + str(accuracy_true) + "%\n")
    f.write("Accuracy on FALSE: " + str(accuracy_false) + "%\n")
    f.write("Percentage of NONE (over the number of incorrect predictions): " + str(percentage_none) + "%\n")
    f.write("Percentage of pure errors (over the number of incorrect predictions): " + str(percentage_pure_errors) + "%\n")
    f.write("Abstention rate (NONE over the total number of predictions): " + str(abstention_rate) + "%\n")
    f.write("Pure error rate (the number of pure errors over the total number of predictions): " + str(pure_error_rate) + "%\n")
    f.write("\n")
    f.close()

# Print the results
print("Script name: " + script_name)
print("Test set file: " + os.path.basename(test_set_file_path))
print("Number of queries: " + str(queries_n))
print("Accuracy: " + str(accuracy) + "%")
print("Macro F1 score: " + str(macro_f1) + "\n")
print("Accuracy on TRUE: " + str(accuracy_true) + "%")
print("Accuracy on FALSE: " + str(accuracy_false) + "%")
print("Percentage of NONE: (over the number of incorrect predictions): " + str(percentage_none) + "%")
print("Percentage of pure errors (over the number of incorrect predictions): " + str(percentage_pure_errors) + "%")
print("Abstention rate (NONE over the total number of predictions): " + str(abstention_rate) + "%")
print("Pure error rate (the number of pure errors over the total number of predictions): " + str(pure_error_rate) + "%")
print("\nResults saved at " + output_file_path)
