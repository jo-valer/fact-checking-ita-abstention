#!/usr/bin/python3

# Usage: python3 create_args_and_run.py <script> <input> <output> <test_set_file> <examples_file>

"""
This script has to create the arguments for the model script, and run it.
"""

import os, sys

SUPPRESS_OUTPUT = True # If True, the output of the running model will be suppressed
SAVE_INFERENCE = True # If True, the inference won't be overwritten

# Read the parameters
if len(sys.argv) != 6:
    print("Usage: python3 create_args_and_run.py <script> <input> <output> <test_set_file> <examples_file>")
    sys.exit(1)

script = sys.argv[1]
input = sys.argv[2]
output = sys.argv[3]
test_set_file = sys.argv[4]
examples_file = sys.argv[5]

# Create the arguments
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
ns = [1, 2, 3, 4, 5]
args_lists = []
for threshold in thresholds:
    for n in ns:
        args = "--threshold " + str(threshold) + " --n " + str(n)
        args_lists.append(args)

# If the results directory does not exist, exit
if not os.path.exists(os.path.join(output, "results")):
    print("Results directory not found")
    sys.exit(1)

# If the inference directory does not exist, exit
if SAVE_INFERENCE:
    if not os.path.exists(os.path.join(output, "inference")):
        print("Inference directory not found")
        sys.exit(1)

output_path = output # Output for the model script only

# Run the script for each set of arguments
for args in args_lists:
    if SAVE_INFERENCE:
        output_path = os.path.join(output, "inference", "semantic_search-" + args.replace("--", "").replace(" ", "-"))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    print("\033[92m" + "Evaluating semantic_search with args: " + args + "\033[0m")
    command = "python3 " + script + " " + input + " --examples " + examples_file + " --output " + output_path + " " + args
    # Run the script
    if SUPPRESS_OUTPUT:
        command += " > /dev/null"
    if os.system(command) != 0:
        print("\033[91m" + "Error in " + script + "\033[0m")
        sys.exit(1)

    # Get results path: output/results/script_name_<variables>.txt (e.g. output_path/results/semantic_search-threshold-0.3-n-1.txt)
    results_path = os.path.join(output, "results", "semantic_search-" + args.replace("--", "").replace(" ", "-") + ".txt")

    # Run compute_metrics.py
    inference = os.path.join(output_path, "inference.tsv")
    command = "python3 compute_metrics.py semantic_search " + test_set_file + " " + inference + " " + results_path
    if os.system(command) != 0:
        print("\033[91m" + "Error in compute_metrics.py" + "\033[0m")
        sys.exit(1)
