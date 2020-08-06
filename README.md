# Masterarbeit

This repository contains all code files relating to my master's thesis.

## Preparing the data sets for the python scripts

- PAWS-X: cat the files `dev_2k.tsv` and `test_2k.tsv` without the first line in the folder `de` into one file `paws-x_de_2k.tsv`:
    - `$ sed -s 1d dev_2k.tsv test_2k.tsv > paws-x_de_2k.tsv`
- SCARE (v1.0.0):
    - cat all .txt-files in the `annotations` folder into one file `annotations.txt`:
        - `$ cat *.txt > annotations.txt`
    - cat all .csv-files in the `annotations` folder into one file `annotations.csv`:
        - `$ cat *.csv > annotations.csv`
- SCARE (v1.0.0_data): cat all .csv files in the `reviews` folder into one file `reviews.csv`:
    - `$ cat *.csv > reviews.csv`
- XNLI: cat the files `xnli.dev.de.tsv` and `xnli.test.de.tsv` in the folder `XNLI-1.0` into one file `xnli.de.tsv`:
    _ `$ cat xnli.dev.de.tsv xnli.test.de.tsv > xnli.de.tsv`
