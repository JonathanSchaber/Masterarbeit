# Masterarbeit

This repository contains all code files relating to my master's thesis.

# Data

The data sets that were used in the experiments, can be obtained from the following sources:

MLQA: [https://github.com/facebookresearch/MLQA](https://github.com/facebookresearch/MLQA)

PAWS-X: [https://github.com/google-research-datasets/paws/tree/master/pawsx](https://github.com/google-research-datasets/paws/tree/master/pawsx)

SCARE: [http://romanklinger.de/scare/](http://romanklinger.de/scare/)

XNLI: [https://cims.nyu.edu/~sbowman/xnli/](https://cims.nyu.edu/~sbowman/xnli/)

XQuAD: [https://github.com/deepmind/xquad](https://github.com/deepmind/xquad)

To reproduce the results, download the data, store them locally, and run the commands under [Preparing the Data](#preparing-the-data)

## Preparing the data

- PAWS-X: cat the files `dev_2k.tsv` and `test_2k.tsv` without the first line in the folder `de` into one file `paws-x_de_2k.tsv`:
    - `$ sed -s 1d dev_2k.tsv test_2k.tsv > paws-x_de_2k.tsv`
- SCARE (v1.0.0): cat all .txt-files in the `annotations` folder into one file `annotations.txt` and cat all .csv-files in the `annotations` folder into one file `annotations.csv`::
    - `$ cat *.txt > annotations.txt && cat *.csv > annotations.csv`
- SCARE (v1.0.0_data): cat all .csv files in the `reviews` folder into one file `reviews.csv`:
    - `$ cat *.csv > reviews.csv`
- XNLI: cat the files `xnli.dev.de.tsv` and `xnli.test.de.tsv` in the folder `XNLI-1.0` into one file `xnli.de.tsv`:
    - `$ cat xnli.dev.de.tsv xnli.test.de.tsv > xnli.de.tsv`
