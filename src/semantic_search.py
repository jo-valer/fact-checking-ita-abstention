#!/usr/bin/python3

# Usage: python3 semantic_search.py input [--examples EXAMPLES_PATH] [--output OUTPUT_PATH] [--n N] [--threshold THRESHOLD]

from sentence_transformers import SentenceTransformer, util
import torch, sys, csv, os, argparse
import pandas as pd

# SETTINGS
ASK_STRING_AS_INPUT = False # if True, the user is asked to confirm when inputting a claim
ASK_OVERWRITE = False # if True, the user is asked to confirm when overwriting an existing file

default_output_path = None
default_examples_path = None

def inference():
    """
    Semantic search to retrieve a label for input claims.
    Load the control set, compute the embeddings, also for the inputs.
    Compute cosine similarity between control set sentences ('examples') and each input claim ('query'), and label the latter with the majority label of the nearest sentences.
    Nearest examples are those N sentences with the highest similarity score, but only if the score is above the threshold.
    In case of ties, the label is set appropriately, according to the logic criteria implemented in the function output_label().
    """
    global default_output_path, default_examples_path

    # Parse the arguments
    args = parse_arguments()

    # Check the arguments
    INPUT_TYPE, INPUT, examples_path, OUTPUT_PATH, N, THRESHOLD = check_args(args)
        
    # Load the dataset
    dataset_path = EXAMPLES_PATH
    dataset = pd.read_csv(dataset_path, sep="\t", header=0, quoting=csv.QUOTE_NONE, dtype={"label": str})

    # Map sentences to embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = dataset['claim'].tolist()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Save embeddings with labels
    corpus_labels = dataset['label'].tolist()

    if INPUT_TYPE == "string":
        query = preprocess_query(INPUT)
        queries = [query]
    elif INPUT_TYPE == "file":
        input_path = INPUT
        with open(input_path, 'r') as f:
            queries = f.readlines()
            queries = [query for query in queries if query.strip() != '']
            queries = [preprocess_query(query) for query in queries]
        
    # Find the closest N sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(N, len(corpus))
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write("output_label\tquery\tmost_similar_examples\n")
        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            similar_claims = []
            print("\n\n======================\n\n")
            print("QUERY:", query)
            print("\nTop " + str(N) + " most similar claims:")
            for score, idx in zip(top_results[0], top_results[1]):
                if score > THRESHOLD:
                    similar_claims.append([corpus[idx], corpus_labels[idx], score.item()])
                    print(corpus[idx], "(Label:", corpus_labels[idx],", Score: {:.3f})".format(score))
                else:
                    # Append [None, NaN] element to keep the same number of elements in the list
                    similar_claims.append([None, None, float('nan')])
            # Label the query with the most frequent label of the retrieved sentences
            similar_claims_labels = [claim[1] for claim in similar_claims]
            majority_label = output_label(similar_claims_labels)
            print("\nLABEL:", majority_label)
            f.write("{}\t{}\t{}\n".format(majority_label, query, similar_claims))
        f.close()
    print("\n\n======================\n\n")
    print("Results are saved in " + OUTPUT_PATH)


def parse_arguments():
    global default_output_path, default_examples_path

    parser = argparse.ArgumentParser()

    # Positional arguments
    parser.add_argument("input", help="Input claim or file containing claims") # Input: single string or file containing claims

    # Optional arguments
    default_examples_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "datasets", "religion_dataset_it", "all.tsv")
    parser.add_argument("--examples", help="Examples dataset", default=default_examples_path) # Examples: tsv file
    default_output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "semantic_search_results")
    parser.add_argument("--output", help="Output folder where to save results", default=default_output_path) # Output: tsv file
    parser.add_argument("--n", help="Number of examples to use (default: 1)", type=int, default=1) # Number of examples to use
    parser.add_argument("--threshold", help="Cosine similarity threshold (default: 0.5)", type=float, default=0.5) # Cosine similarity threshold

    args = parser.parse_args()

    return args


def check_args(args):
    """
    Check the arguments and return input type (string or file), input, examples path, output path, N and threshold
    """
    global ASK_STRING_AS_INPUT, ASK_OVERWRITE, default_output_path

    # Check if input is a file
    if not os.path.isfile(args.input):
        input_type = "string"
        # Ask user if he wants to continue
        print("\033[93mWARNING: Input is not a file. It will be considered a string.\033[0m")
        if ASK_STRING_AS_INPUT:
            answer = input("Do you want to continue? [y/n] ")
            if answer.lower in ['n', 'no']:
                print("Exiting...")
                exit(0)
    else:
        input_type = "file"

    # Check if examples_path exists as a file and is a tsv file
    if not os.path.isfile(args.examples):
        print("Error: examples file does not exist.")
        sys.exit(1)
    if not args.examples.endswith(".tsv"):
        print("Error: examples file is not a tsv file")
        sys.exit(1)

    # Check if output_path already exists
    if not os.path.isdir(args.output): # It doesn't exist
        if args.output != default_output_path: # Not default: error
            print("Error: " + args.output + " is not a directory")
            sys.exit(1)
        else: # Default: create the output directory
            print("Creating output directory " + args.output)
            os.makedirs(args.output, exist_ok=True)
    output_file_path = os.path.join(args.output, "inference.tsv")
    if os.path.isfile(output_file_path):
        # Ask user if he wants to overwrite the file
        print("\033[93mWARNING: Output path already exists. It will be overwritten.\033[0m")
        if ASK_OVERWRITE:
            answer = input("Do you want to continue? [y/n] ")
            if answer.lower in ['n', 'no']:
                print("Exiting...")
                exit(0)

    return input_type, args.input, args.examples, output_file_path, args.n, args.threshold


def preprocess_query(query):
    """
    Remove tabs and newlines from the query.
    """
    query = query.replace("\t", " ") # Replace tabs with spaces
    query = query.replace("\n", "") # Remove newlines
    return query


def output_label(labels):
    """
    Return the most frequent label in the list of labels.
    If there is a tie, return the label according to a logic criteria:
    - TRUE==FALSE -> HALF-TRUE
    - TRUE==HALF-TRUE -> TRUE
    - FALSE==HALF-TRUE -> FALSE
    If specified in the settings, HALF-TRUE labels are considered as FALSE.
    If all labels are None, return None.
    """
    HALF_TRUE_AS_FALSE = True # If True, HALF-TRUE labels are considered as FALSE
    
    # Count occurrences of each label
    labels=[label.lower() if label!=None else label for label in labels]
    trues = labels.count('true')
    falses = labels.count('false')
    half_trues = labels.count('half-true')
    nones = labels.count('none')
    print(trues,falses,half_trues,nones)
    if trues+falses+half_trues+nones!=len(labels):
        print("\033[93mWARNING: Some labels not recognized:\033[0m",set(labels))
    if HALF_TRUE_AS_FALSE:
        falses += half_trues
        half_trues = 0

    # If all labels are None, return None
    if trues == 0 and falses == 0 and half_trues == 0:
        return None
    
    # if the number of None labels is greater than the number of other labels, return None
    if nones > trues+falses+half_trues:
        return None
    
    # Label the query with the most frequent label
    if trues > falses and trues > half_trues:
        output_label = "true"
    elif falses > trues and falses > half_trues:
        output_label = "false"
    elif half_trues > trues and half_trues > falses:
        output_label = "half-true"
    elif trues == falses and trues > half_trues:
        output_label = "half-true"
    elif trues == half_trues and trues > falses:
        output_label = "true"
    elif falses == half_trues and falses > trues:
        output_label = "false"
    else:
        output_label = None
    
    return output_label



# EXECUTION

inference()
