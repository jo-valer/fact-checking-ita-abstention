#!/usr/bin/python3

# Usage: python3 get_stats.py <directory> <output_folder>

"""
This script has to get the stats from the specified directory.
Each file in directory/results is supposed to be a result file, named <model>-<args>.txt. E.g. semantic_search-threshold-0.35-n-6.txt
The script has to get the accuracy percentage from each file, and print it in a table.
"""

import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TITLE = False # Set to False in order not to add a title to the plots
UNNECESSARY_IMAGES = False # Set to False in order to avoid creating plots not needed for the paper

# Read the parameters
if len(sys.argv) != 3:
    print("Usage: python3 get_stats.py <directory> <output_folder>")
    sys.exit(1)

parent_directory = sys.argv[1]
output_folder = sys.argv[2]

output_directory = os.path.join(output_folder)
# Delete the previous results
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.makedirs(output_directory)

# Get last two folders of the path as title
title = str(os.path.basename(os.path.dirname(output_folder))) + "/" + str(os.path.basename(output_folder))
# If the title ends with slash, remove it
if title.endswith("/"):
    title = title[:-1]

# Get the list of tests (i.e. names of the subdiretories in the parent directory)
tests = os.listdir(parent_directory)

# PLOTS FOR EACH TEST (news-like, social-like)
for test in tests:
    print("\033[92mTest-set: " + test + "\033[0m")
    directory = os.path.join(parent_directory, test)
    results_directory = os.path.join(directory, "results")

    # Check if the directory exists
    if not os.path.isdir(directory):
        print("Directory not found: " + directory)
        sys.exit(1)
    
    # if not os.path.isdir(os.path.join(output_folder)):
    #     os.makedirs(os.path.join(output_folder))

    # Get the list of files in the directory
    files = os.listdir(results_directory)

    # Keep only the files that end with .txt
    files = [file for file in files if file.endswith(".txt")]

    # Get the list of models
    models = ["semantic_search"]

    table = [] # List of lists, each list is a row of the table

    # Matrix of semantic_search results, rows are thresholds, columns are ns. Each cell is a list of accuracies. Notice that threshold and n are floats and ints, respectively.
    semantic_search_thresholds = []
    semantic_search_ns = []
    semantic_search_results = [] # Arrary of accuracies
    accuracy_matrix = [] # Matrix of accuracies (rows in the number of Ns, columns in the number of thresholds)
    f1_score_matrix = [] # Matrix of F1 scores (rows in the number of Ns, columns in the number of thresholds)
    nones_matrix = [] # Matrix of percentages of Nones (rows in the number of Ns, columns in the number of thresholds)
    pure_errors_matrix = [] # Matrix of percentages of pure errors (rows in the number of Ns, columns in the number of thresholds)
    abstention_rate_matrix = [] # Matrix of abstention rates (rows in the number of Ns, columns in the number of thresholds)
    pure_error_rate_matrix = [] # Matrix of pure error rates (rows in the number of Ns, columns in the number of thresholds)

    # Check all file names to get the list of thresholds and ns
    for file in files:
        args = file.split("-")
        model = args[0]
        if model not in models:
            print("Model not recognized:")
            print(file, model)
        elif model == "semantic_search":
            threshold = args[2]
            n = args[4].split(".")[0]
            args = "--threshold " + threshold + " --n " + n
            threshold = float(threshold)
            n = int(n)
            if threshold not in semantic_search_thresholds:
                semantic_search_thresholds.append(threshold)
            if n not in semantic_search_ns:
                semantic_search_ns.append(n)

    # Reiterate over the files to get the accuracies
    for file in files:
        # Get the accuracy
        with open(os.path.join(results_directory, file), "r") as f:
            for line in f:
                if line.startswith("Accuracy:"):
                    accuracy = float(line.split(":")[1].split("%")[0])
                    semantic_search_results.append(accuracy)
                if line.startswith("Macro F1 score:"): # be aware there is a space after the colon, but no percentage sign
                    f1_score = float(line.split(":")[1])
                if line.startswith("Percentage of NONE (over the number of incorrect predictions):"):
                    none_percentage = float(line.split(":")[1].split("%")[0])
                if line.startswith("Percentage of pure errors (over the number of incorrect predictions):"):
                    pure_error_percentage = float(line.split(":")[1].split("%")[0])
                if line.startswith("Abstention rate (NONE over the total number of predictions):"):
                    abstention_rate = float(line.split(":")[1].split("%")[0])
                if line.startswith("Pure error rate (the number of pure errors over the total number of predictions):"):
                    pure_error_rate = float(line.split(":")[1].split("%")[0])
        args = file.split("-")
        model = args[0]
        if model == "semantic_search":
            threshold = args[2]
            n = args[4].split(".")[0]
            args = "--threshold " + threshold + " --n " + n
            threshold = float(threshold)
            n = int(n)
            # Update the matrix
            if len(accuracy_matrix) == 0:
                for i in range(len(semantic_search_ns)):
                    accuracy_matrix.append([])
                    f1_score_matrix.append([])
                    nones_matrix.append([])
                    pure_errors_matrix.append([])
                    abstention_rate_matrix.append([])
                    pure_error_rate_matrix.append([])
                for i in range(len(semantic_search_ns)):
                    for j in range(len(semantic_search_thresholds)):
                        accuracy_matrix[i].append([])
                        f1_score_matrix[i].append([])
                        nones_matrix[i].append([])
                        pure_errors_matrix[i].append([])
                        abstention_rate_matrix[i].append([])
                        pure_error_rate_matrix[i].append([])
            # Update the matrix
            idx_n = semantic_search_ns.index(n)
            idx_threshold = semantic_search_thresholds.index(threshold)
            accuracy_matrix[idx_n][idx_threshold].append(accuracy)
            f1_score_matrix[idx_n][idx_threshold].append(f1_score)
            nones_matrix[idx_n][idx_threshold].append(none_percentage)
            pure_errors_matrix[idx_n][idx_threshold].append(pure_error_percentage)
            abstention_rate_matrix[idx_n][idx_threshold].append(abstention_rate)
            pure_error_rate_matrix[idx_n][idx_threshold].append(pure_error_rate)
        # print(model + "\t" + args + "\t" + str(accuracy))
        table.append([model, args, accuracy])


    # LOG FILES =======================================================================================================

    log_file = os.path.join(output_folder, test + "_log.txt")

    matrices = [accuracy_matrix, nones_matrix, pure_errors_matrix, abstention_rate_matrix, pure_error_rate_matrix, f1_score_matrix]

    with open(log_file, "w") as f:
        # Clustering results
        f.write("Test-set: " + test + "\n\n")

        for matrix in matrices:
            f.write("\n")
            if matrix == accuracy_matrix:
                f.write("Accuracy matrix\n")
                print("\nAccuracy matrix")
            if matrix == f1_score_matrix:
                f.write("Macro F1 score matrix\n")
                print("\nMacro F1 score matrix")
            if matrix == nones_matrix:
                f.write("Percentage of NONE (over the number of incorrect predictions) matrix\n")
                print("\nPercentage of NONE (over the number of incorrect predictions) matrix")
            if matrix == pure_errors_matrix:
                f.write("Percentage of pure errors (over the number of incorrect predictions) matrix\n")
                print("\nPercentage of pure errors (over the number of incorrect predictions) matrix")
            if matrix == abstention_rate_matrix:
                f.write("Abstention rate (NONE over the total number of predictions) matrix\n")
                print("\nAbstention rate (NONE over the total number of predictions) matrix")
            if matrix == pure_error_rate_matrix:
                f.write("Pure error rate (the number of pure errors over the total number of predictions) matrix\n")
                print("\nPure error rate (the number of pure errors over the total number of predictions) matrix")

            header = ""
            for threshold in semantic_search_thresholds:
                # If the threshold has only one decimal (e.g. 0.5), add a 0 at the end (e.g. 0.50), but if it has two decimals (e.g. 0.55), leave it as it is
                if len(str(threshold).split(".")[1]) == 1:
                    header += "\t" + str(threshold) + "0"
                else:
                    header += "\t" + str(threshold)
            f.write(header + "\n")
            print(header)
            for i in range(len(semantic_search_ns)):
                row = str(semantic_search_ns[i])
                for j in range(len(semantic_search_thresholds)):
                    row += "\t" + str(np.mean(matrix[i][j]))
                f.write(row + "\n")
                print(row)
        
        f.close()

    # TSV FILES =======================================================================================================
    # The TSV files are used to plot the results with Pyplot

    tsv_files = ["accuracy.tsv", "nones.tsv", "pure_errors.tsv", "abstention_rate.tsv", "pure_error_rate.tsv", "f1_score.tsv"]

    for i in range(len(tsv_files)):
        tsv_files[i] = os.path.join(output_directory, tsv_files[i])
    
    for i in range(len(matrices)):
        with open(tsv_files[i], "w") as f:
            f.write("n\tthreshold\tvalue\n")
            for j in range(len(semantic_search_ns)):
                for k in range(len(semantic_search_thresholds)):
                    f.write(str(semantic_search_ns[j]) + "\t" + str(semantic_search_thresholds[k]) + "\t" + str(matrices[i][j][k])[1:-1] + "\n")
            f.close()


# Zip the output folder
shutil.make_archive(output_directory, "zip", output_directory)
print("Output folder zipped in " + output_directory + ".zip\n")
