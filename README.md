# Masterarbeit

This repository contains all code files relating to my master's thesis.

To reproduce the results, run the commands under [Preparing the Data](#preparing-the-data)

# Architecture

![GLIBERT Architecture](/thesis/clvorlage/images/architecture.png)

# Data

Overview of the data sets used in the experiments.

| Data Set | NLP Task | ML Task | \# Examples | Splits |
| -------- | -------- | ------- | ----------- | ------ |
| [deISEAR](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/deisear/) |  Emotion Detection | Multi-Class Classification  | 1,001 | - |
| [MLQA](https://github.com/facebookresearch/MLQA) | Question Answering | Span Prediction | 509/4,499 | dev/test |
| [PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx) | Paraphrase Identification | Binary Classification | 49,402/2,000/2,000 | train/dev/test |
| [SCARE](http://romanklinger.de/scare/) | Sentiment Analysis | Multi-Class Classification | 1,760 | - |
| [SCARE Reviews](http://romanklinger.de/scare/) |  Sentiment Analysis | Multi-Class Classification | 802,860 | - |
| [XNLI](https://cims.nyu.edu/~sbowman/xnli/) | Natural Language Inference | Multi-Class Classification |  2,489/7,498 | dev/test |
| [XQuAD](https://github.com/deepmind/xquad) | Question Answering | Span Prediction |  1,192 | - |

# Results

# Preparing the Data

Download the data and adhere to the following folder structure (it is important that the names of the directories are exactly the same as below; e.g. «deISEAR» and not «deISEARenISEAR» as it is named when downloaded etc. The structure of the downloaded data sets is not changed, so naming the root directory correctly is the main thing to follow):

	/<path>/<to>/deISEAR/  
			../deISEAR.tsv
	
	
	/<path>/<to>/MLQA/
			../dev/dev-context-de-question-de.tsv
			../test/test-context-de-question-de.json
	
	/<path>/<to>/PAWS-X/
			../de/
				../dev_2k.tsv
				../test_2k.tsv	
				../translated_train.tsv
	
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

When you obtained the .txt files, move them to the folder «annotations», where the other files (.csv etc.) lie.

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


# SRL Resources

## ParZu

Clone the [repository](https://github.com/rsennrich/ParZu/) and configure
config.ini file to run locally on your machine.

## DAMESRL

Download the source code from [here](https://liir.cs.kuleuven.be/software_pages/damesrl.php)
in a local folder DAMESRL``, and configure `server_configs/charatt_ger_pred_infer.ini` to run
locally on your machine.


`$ export PYTHONPATH="${PYTHONPATH}:/<path>/<to>/DAMESRL"`

`$ export PYTHONPATH="${PYTHONPATH}:/<path>/<to>/ParZu"`
