# Automated Fact-checking in Italian Under Domain Shift

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains code and data associated with the CLiC-it 2023 paper:

Giovanni Valer, Alan Ramponi and Sara Tonelli. 2023. **When You Doubt, Abstain: A Study of Automated Fact-checking in Italian Under Domain Shift**. In *Proceedings of the Ninth Italian Conference on Computational Linguistics*, Venice, Italy. CEUR.org. [[cite]](#citation) [[paper]](https://ceur-ws.org/Vol-3596/paper50.pdf)


## Getting started

To get started, clone this repository on your own path:
```
git clone https://github.com/jo-valer/fact-checking-ita-abstention.git
```


### Environment

Create an environment with your own preferred package manager. We used [python 3.9.13](https://www.python.org/downloads/release/python-3913/) and dependencies listed in [`requirements.txt`](requirements.txt). If you use [conda](https://docs.conda.io/en/latest/), you can just run the following commands from the root of the project:

```
conda create --name fact-checking-ita-abstention python=3.9.13    # create the environment
conda activate fact-checking-ita-abstention                       # activate the environment
pip install --user -r requirements.txt                            # install the required packages
```


### Data

In the `data/controlsets/` folder is the X-Fact dataset [(Gupta and Srikumar, 2021)](https://aclanthology.org/2021.acl-short.86/) [[repository]](https://github.com/utahnlp/x-fact), already filtered to contain examples in Italian only and without instances labeled as *complicated*. The resulting files are described below, using the notation introduced in the paper:
- `train.tsv`: TRAIN
- `train_dev.tsv`: TRAIN + DEV
- `train_dev_id.tsv`: TRAIN + DEV + TEST<sub>id</sub>
- `train_dev_id_ood.tsv`: TRAIN + DEV + TEST<sub>id</sub> + TEST<sub>ood</sub>
- `train_dev_ood.tsv`: TRAIN + DEV + TEST<sub>ood</sub>

In the `data/testsets/` folder are the challenge test sets, annotated according to our claim ambiguity categorization (see the [paper](https://ceur-ws.org/Vol-3596/paper50.pdf) for more details). The columns `news-like` and `social-like` contain the rewritten versions of the original claim (i.e., column `claim`), whereas the `ambiguity` column indicates the ambiguity label for the claim. The files are the following:
- `in_domain.tsv`: _in-domain_ test set (all genres: original, news-like, and social-like)
- `out_of_domain.tsv`: _out-of-domain_ test set (all genres: original, news-like, and social-like) 


## Replicating the experiments

Run the experiments in a ***controlled** setup*:
```
cd src
./experiments.sh . ../data/testsets/ ../data/controlsets/ 1     # in-domain test
./experiments.sh . ../data/testsets/ ../data/controlsets/ 3     # out-of-domain test
```
Run the experiments in a ***non-controlled** setup*:
```
cd src
./experiments.sh . ../data/testsets/ ../data/controlsets/ 0     # in-domain test
./experiments.sh . ../data/testsets/ ../data/controlsets/ 2     # out-of-domain test
```

The results are saved in the `results/` folder.


## Citation

If you use or build on top of this work, please cite our paper as follows:

```
@inproceedings{valer-etal-2023-when,
    title={When You Doubt, Abstain: {A} Study of Automated Fact-checking in {I}talian Under Domain Shift},
    author={Valer, Giovanni and Ramponi, Alan and Tonelli, Sara},
    booktitle={Proceedings of the 9th Italian Conference on Computational Linguistics},
    publisher={CEUR-ws.org},
    year={2023},
    month={november},
    address={Venice, Italy}
}
```
