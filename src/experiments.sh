#!/usr/bin/bash

# Usage: ./experiments.sh <model_folder> <test_set_folder> <examples_file_folder> <mode>

# Check arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: ./experiments.sh <model_folder> <test_set_folder> <examples_file_folder> <mode>"
    exit 1
fi

model_script="$1/semantic_search.py"
# Check if model folder contains the model
if [ ! -f "$model_script" ]; then
    echo "Model script not found"
    exit 1
fi

in_domain_set_file="$2/in_domain.tsv"
ood_set_file="$2/out_of_domain.tsv"
# Check if test set folder contains the test-sets
if [ ! -f "$in_domain_set_file" ]; then
    echo "Test set not found"
    exit 1
fi
if [ ! -f "$ood_set_file" ]; then
    echo "Out-of-domain set not found"
    exit 1
fi

# Check if examples file folder exists
if [ ! -d "$3" ]; then
    echo "Examples file folder not found"
    exit 1
fi

# GET VARIBLES ---------------------------------------------

# Get full path of the test sets files
in_domain_set_file=$(realpath $in_domain_set_file)
# Get full path of the out-of-domain set file
ood_set_file=$(realpath $ood_set_file)
# Get full path of the examples file
examples_file_folder=$(realpath $3)

alpha1="idtestset"
alpha2="oodtestset"

# Otput path is the folder of this script
output_path=$(dirname $(realpath $0))
# Get full output path
output_path=$(realpath $output_path)

inference=$output_path/inference.tsv

# The results path is in the parent folder of this script
results_path=$(dirname $output_path)
results_path=$results_path/results

# ----------------------------------------------------------

# The mode is a number identifying the type of experiment to run
# 0: NON-CONTROLED SETUP: control-set = train + dev; test-sets = in-domain (claim, social-like)
# 1: CONTROLED SETUP: control-set = train + dev + in-domain; test-sets = in-domain (news-like, social-like)
# 2: NON-CONTROLED SETUP: control-set = train + dev; test-sets = out-of-domain (claim, social-like)
# 3: CONTROLED SETUP: control-set = train + dev + out-of-domain; test-sets = out-of-domain (news-like, social-like)
# 4: NON-CONTROLED SETUP: control-set = train + dev + in-domain; test-sets = out-of-domain (claim, social-like)

if [ $4 -eq 0 ]; then
    mode="mode0"
    results_name="non-controlled/in-domain"
    columns=("claim" "social-like")
    test_sets_list=($alpha1)
    examples_file=$examples_file_folder/train_dev.tsv
elif [ $4 -eq 1 ]; then
    mode="mode1"
    results_name="controlled/in-domain"
    columns=("news-like" "social-like")
    test_sets_list=($alpha1)
    examples_file=$examples_file_folder/train_dev_id.tsv
elif [ $4 -eq 2 ]; then
    mode="mode2"
    results_name="non-controlled/out-of-domain"
    columns=("claim" "social-like")
    test_sets_list=($alpha2)
    examples_file=$examples_file_folder/train_dev.tsv
elif [ $4 -eq 3 ]; then
    mode="mode3"
    results_name="controlled/out-of-domain"
    columns=("news-like" "social-like")
    test_sets_list=($alpha2)
    examples_file=$examples_file_folder/train_dev_ood.tsv
else
    echo "Mode not valid"
    exit 1
fi

# Delete the previous results folder
rm -rf $output_path/$mode

for column in ${columns[@]}; do
    echo -e "\e[32m=============================================================\e[0m"
    echo -e "\e[32mCOLUMN: $column\e[0m"
    echo
    
    for test_set in ${test_sets_list[@]}; do
        # Get full path of the test set file
        if [ $test_set == $alpha1 ]; then
            test_set_file=$in_domain_set_file
        else
            test_set_file=$ood_set_file
        fi
        # Get specific output path
        specific_output_path=$output_path/$mode/$test_set/$column

        # Create the results folder
        mkdir -p $specific_output_path
        mkdir $specific_output_path/results
        mkdir $specific_output_path/inference

        # Run test_to_input.py
        echo -e "\e[32mCreating the input...\e[0m"
        echo -e "\e[32mTest set: $test_set\e[0m"
        python3 test_to_input.py $test_set_file $column $specific_output_path
        # Check the exit code of the previous command, if it is not 0, exit the script
        if [ $? -ne 0 ]; then
            echo -e "\e[31mError in test_to_input.py\e[0m"
            exit 1
        fi

        # Using create_args_and_run.py, run the models with different arguments
        echo -e "\e[32mRunning the model...\e[0m"
        python3 create_args_and_run.py $model_script $specific_output_path/input.txt $specific_output_path $test_set_file $examples_file
        # Check the exit code of the previous command, if it is not 0, exit the script
        if [ $? -ne 0 ]; then
            echo -e "\e[31mError in create_args_and_run.py\e[0m"
            exit 1
        fi
    done
done

for test_set in ${test_sets_list[@]}; do
    echo -e "\e[32m=============================================================\e[0m"
    echo -e "\e[32mCreating stats for: $test_set\e[0m"
    echo
    # Run get_stats.py
    python3 get_stats.py $output_path/$mode/$test_set $results_path/$results_name
    # Check the exit code of the previous command, if it is not 0, exit the script
    if [ $? -ne 0 ]; then
        echo -e "\e[31mError in get_stats.py\e[0m"
        exit 1
    fi
done
