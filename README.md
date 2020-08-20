# Masterarbeit

This repository contains all code files relating to my master's thesis.

To reproduce the results, run the commands under [Preparing the Data](#preparing-the-data)

# Architecture

![GLIBERT Architecture](/thesis/clvorlage/images/architecture.png)

# Data

The data sets that were used in the experiments, can be obtained from the following sources:

MLQA: [https://github.com/facebookresearch/MLQA](https://github.com/facebookresearch/MLQA)

PAWS-X: [https://github.com/google-research-datasets/paws/tree/master/pawsx](https://github.com/google-research-datasets/paws/tree/master/pawsx)

SCARE: [http://romanklinger.de/scare/](http://romanklinger.de/scare/)

XNLI: [https://cims.nyu.edu/~sbowman/xnli/](https://cims.nyu.edu/~sbowman/xnli/)

XQuAD: [https://github.com/deepmind/xquad](https://github.com/deepmind/xquad)


| Data Set | NLP Task | ML Task | \# Examples | Splits |
| -------- | -------- | ------- | ----------- | ------ |
| deISEAR |  Emotion Detection | Multi-Class Classification  | 1 001 | - |
| MLQA | Question Answering | Span Prediction | 509/4 499 | dev/test |
| PAWS-X | Paraphrase Identification | Binary Classification | 2 000/4 000 | dev/test |
| SCARE | Sentiment Analysis | Multi-Class Classification | 1 760 | - |
| SCARE Reviews |  Sentiment Analysis | Multi-Class Classification | 802 860 | - |
| XNLI | Natural Language Inference | Multi-Class Classification |  2 489/7 498 | dev/test |
| XQuAD | Question Answering | Span Prediction |  1 192 | - |

# Results

# Preparing the Data


Download the data and adhere to the following folder structure (it is important that the names of the root directory names are exactly the same as below; e.g. «deISEAR» and not «deISEARenISEAR» etc.):

	/<path>/<to>/deISEAR/  
			..deISEAR.tsv
	
	
	/<path>/<to>/MLQA/
			../dev/dev-context-de-question-de.tsv
			../test/test-context-de-question-de.json
	
	/<path>/<to>/PAWS-X/
			../de/
				../dev_2k.tsv
				../test_2k.tsv	
	
	/<path>/<to>/SCARE/
			../scare_v1.0.0/annotations/
				../alarm_clocks.csv
				../alarm_clocks.rel
				../alarm_clocks.txt
				../...
			../scare_v1.0.0_data/reviews/
				../alarm_clocks.csv
				../...
	
	/<path>/<to>/XNLI/
			../XNLI-1.0/
				../xnli.dev.de.tsv
				../xnli.test.de.tsv
	
	/<path>/<to>/XQuAD/
			../xquad/
				../xquad.de.json


## Pre-Preprocessing

Run the following bash commands to prepare the data sets for the python scripts:

#### SCARE

When you obtained the .txt files, move them to the folder «annotations», where the other file (.csv etc.) lie.

cd into the folder «annotations»:

`$ cd /<path>/<to>/SCARE/scare_v1.0.0/annotations/`

cat all the .txt and .csv files into one file, respectively:

`$ cat *.txt > annotations.txt && cat *.csv > annotations.csv`

#### SCARE Reviews

cd into the folder «reviews»:

`$ cd /<path>/<to>/SCARE/scare_v1.0.0_data/reviews/`

cat all the .csv files into one .csv files:

`$ cat *.csv > reviews.csv`

## Preprocessing

For each data set, run the following command:

	python preprocess_data.py \
		-d <data set> \
		-p /<path>/<to>/<data set>/ \
		-a /<path>/<to>/DAMESRL/server_configs/srl_char_att_ger_infer.ini
`

# SRL Resources
