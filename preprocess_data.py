import argparse
import csv
import json
import os
import re

from pathlib import Path
from predict_SRL import SRL_predictor
from random import shuffle


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-a", 
            "--argument_model_config", 
            type=str, 
            help="Argument model config for DAMESRL",
            )
    parser.add_argument(
            "-b", 
            "--bert_path", 
            type=str, 
            help="Path to BERT model",
            )
    parser.add_argument(
            "-d", 
            "--data_set", 
            type=str, 
            help="Indicate on which data set model should be preprocessed",
            choices=["deISEAR", "XNLI", "SCARE", "PAWS-X", "MLQA", "XQuAD"]
            )
    parser.add_argument(
            "-p", 
            "--path", 
            type=str, 
            help="Path to directory containing the files",
            )
    return parser.parse_args()


def write_to_files(data, files):
    counter = 0
    len_train = int(len(data)*0.7)
    len_dev = int(len(data)*0.15)
    shuffle(data)

    print("======== Writing to files: ========")
    print("{}\n{}\n{}".format(files[0], files[1], files[1]))

    with open(files[0], "w") as f:
        for i, element in enumerate(data[:len_train]):
            csv.writer(f, delimiter="\t").writerow([i]+element)
        counter += i
    with open(files[1], "w") as f:
        for i, element in enumerate(data[len_train:len_train+len_dev]):
            i += counter
            csv.writer(f, delimiter="\t").writerow([i+counter]+element)
    with open(files[2], "w") as f:
        for i, element in enumerate(data[len_train+len_dev:]):
            i += counter
            csv.writer(f, delimiter="\t").writerow([i+counter]+element)


def splitted_write_to_files(data, files, i):
    if i == 0:
        len_train = int(len(data)*0.85)
        counter = 0

        print("======== Writing to file: {} ========".format(files[0]))
        with open(files[0], "w") as f:
            for j, element in enumerate(data[:len_train]):
                csv.writer(f, delimiter="\t").writerow([j]+element)
            counter += j
        print("======== Writing to file: {} ========".format(files[1]))
        with open(files[1], "w") as f:
            for j, element in enumerate(data[len_train:]):
                j += counter
                csv.writer(f, delimiter="\t").writerow([j]+element)
    else:
        print("======== Writing to file: {} ========".format(files[i]))
        with open(files[1], "r") as f:
            lst = [x for x in csv.reader(f, delimiter="\t")]
            counter = eval(lst[-1][0]) + 1

        with open(files[2], "w") as f:
            for j, element in enumerate(data):
                j += counter
                csv.writer(f, delimiter="\t").writerow([j]+element)


def get_majority_label(labels):
    """function for exctracting majority label out of n labels.
    Used for SCARE data set preprocessing
    Args:
        param1: list[str]
    Returns:
        tuple[str, bool, int)
    """
    pos = 0
    neg = 0
    num = len(labels)

    for label in labels:
        if label.strip() == "Positive":
            pos += 1
        elif label.strip() == "Negative":
            neg += 1
        else:
            continue

    if len(labels) == 0:
        return ("Neutral", False, num)
    elif pos == neg:
        return ("Neutral", 0, num)
    elif pos > neg:
        margin = int(((pos - neg)**2)**0.5)
        return ("Positive", True, num) if margin > 1 else ("Positive", False, num)
    else:
        margin = int(((pos - neg)**2)**0.5)
        return ("Negative", True, num) if margin > 1 else ("Negative", False, num)


def preprocess_deISEAR(path):
    """Preprocess deISEAR data
    Preprocesses the deISEAR data set.
    Writes dev and test files.
    Args:
        param1: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/deISEARenISEAR/, not file!"
    path = str(path)
    file_path = path + "/deISEAR.tsv"
    outfile_paths = [
            path + "/gliBert_deISEAR_train.tsv",
            path + "/gliBert_deISEAR_dev.tsv",
            path + "/gliBert_deISEAR_test.tsv"
            ]

    emotion_sentence_srl = []

    with open(file_path, "r") as f:
        f_reader = csv.reader(f, delimiter="\t")
        next(f_reader)
        for i, row in enumerate(f_reader):
            emotion, sentence = row[1], row[2]
            sentence = re.sub("\.\.\.", "[MASK]", sentence)
            sem_roles = srl_predictor.predict_semRoles(re.sub("[MASK]", "Angst", sentence))
            if sem_roles == False:
                continue
            emotion_sentence_srl.append([emotion, "", sentence, sem_roles])

    write_to_files(emotion_sentence_srl, outfile_paths)


def preprocess_MLQA(path):
    """Preprocess MLQA data
    Preprocesses the MLQA data set.
    Writes dev and test files.
    Args:
        param1: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/MLQA/, not file!"
    path = str(path)
    file_paths = [
            path + "/dev/dev-context-de-question-de.json",
            path + "/test/test-context-de-question-de.json"
            ]
    outfile_paths = [
            path + "/gliBert_mlqa_train.tsv",
            path + "/gliBert_mlqa_dev.tsv",
            path + "/gliBert_mlqa_test.tsv"
            ]

    too_long_contexts = []

    for h, file_path in enumerate(file_paths):
        spans_text_qas_srl = []
        with open(file_path, "r") as f:
            file = f.read()
            json_data = json.loads(file)
        
        for i in range(len(json_data["data"])):
            for j in range(len(json_data["data"][i]["paragraphs"])):
                context = json_data["data"][i]["paragraphs"][j]["context"]
                try:
                    sem_roles_context = srl_predictor.predict_semRoles(context)
                except:
                    too_long_contexts.append(context)
                    continue
                for k in range(len(json_data["data"][i]["paragraphs"][j]["qas"])):
                    question = json_data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                    start_index = json_data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["answer_start"]
                    text = json_data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                    sem_roles_question = srl_predictor.predict_semRoles(question)
                    if sem_roles_question == False or sem_roles_context == False:
                        continue
                    spans_text_qas_srl.append([
                                            start_index,
                                            text,
                                            context,
                                            question,
                                            sem_roles_context,
                                            sem_roles_question
                                            ])

        splitted_write_to_files(spans_text_qas_srl, outfile_paths, h)

    with open(path + "/too_long.txt", "w") as f:
        for context in too_long_contexts:
            f.write(context + "\n\n")


def preprocess_PAWS_X(path):
    """read in merged TSVs, predict SRLs, write label, text and SRLs to new file
    ATTENTION: path points to directory, not input file!
    Args:
        param1: str
        param2: str
        param3: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/PAWS-X/, not file!"
    path = str(path)
    file_paths = [
            path + "/de/translated_train.tsv",
            path + "/de/dev_2k.tsv",
            path + "/de/test_2k.tsv"
            ]
    outfile_paths = [
            path + "/gliBert_paws_x_train.tsv",
            path + "/gliBert_paws_x_dev.tsv",
            path + "/gliBert_paws_x_test.tsv"
            ]

    counter = 0
    for i, file_path in enumerate(file_paths):
        label_text_feat = []
        with open(file_path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            next(f_reader)
            for j, row in enumerate(f_reader):
                if j % 100 == 0:
                    print("Processing example {} out of 49 400.".format(j))
                try:
                    para_id, sentence_1, sentence_2, label = row[0], row[1], row[2], row[3]
                except:
                    print("ERROR:")
                    print("Could not parse line, Id: {}. Skipping.")
                sem_roles_1 = srl_predictor.predict_semRoles(sentence_1)
                sem_roles_2 = srl_predictor.predict_semRoles(sentence_2)
                if sem_roles_1 == False or sem_roles_2 == False:
                    continue
                elif sentence_1 == "NS" or sentence_2 == "NS":
                    print("ERROR:")
                    print("Undefined Sentence found. Id: {}. Skipping.".format(para_id))
                    continue
                label_text_feat.append([label, "", sentence_1, sentence_2, sem_roles_1, sem_roles_2])

            with open(outfile_paths[i], "w") as f:
                for j, element in enumerate(label_text_feat):
                    j += counter
                    csv.writer(f, delimiter="\t").writerow([j]+element)
                counter += j


def preprocess_SCARE(path):
    """read in merged TSVs, predict SRLs, write label, text and SRLs to new file
    ATTENTION: path points to directory, not input file!
    Args:
        param1: str
        param2: str
        param3: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/SCARE/, not file!"
    path = str(path)
    id_text_labels = {}
    label_text_feat = []
    outfile_paths = [
            path + "/gliBert_scare_annotations_train.tsv",
            path + "/gliBert_scare_annotations_dev.tsv",
            path + "/gliBert_scare_annotations_test.tsv"
            ]

    count_non_maj = 0
    count_no_labels = 0
    count_close = 0
    count_all = 0

    with open(path + "/scare_v1.0.0/annotations/annotations.txt", "r") as f:
        ids_texts = [example for example in csv.reader(f, delimiter="\t")]
    with open(path + "/scare_v1.0.0/annotations/annotations.csv", "r") as f:
        rows = csv.reader(f, delimiter="\t")
        ids_labels = []
        for row in rows:
            entity, review_id, left, right, string, phrase_id, polarity, relation = row
            # since aspects are always neutral, we can ignore them
            if entity == "subjective":
                ids_labels.append([review_id, polarity])

    for review_id, text in ids_texts:
        if review_id in id_text_labels:
            raise Error
        else:
            id_text_labels[review_id] = {"text": text, "labels": []}

    for review_id, label in ids_labels:
        if review_id not in id_text_labels:
            raise Error
        else:
            id_text_labels[review_id]["labels"].append(label)

    for review_id, feat in id_text_labels.items():
        polarity, majority, num_labels = get_majority_label(feat["labels"])
        sem_roles = srl_predictor.predict_semRoles(feat["text"])
        if sem_roles == False:
            continue
        label_text_feat.append([polarity, "", feat["text"], sem_roles])
        if polarity == "Neutral": count_non_maj += 1 
        if not majority: count_close += 1 
        if num_labels == 0:
            count_no_labels += 1
        else:
            count_all += num_labels 
    
    print("")
    print("======== Stats ========")
    print("{} reviews had no labels".format(count_no_labels))
    print("{:.2f}% of votes were non-majority".format(count_non_maj/count_all*100))
    print("{:.2f}% of votes were close (label difference of 1)".format(count_close/count_all*100))

    write_to_files(label_text_feat, outfile_paths)


def preprocess_SCARE_reviews(path, path_outfile):
    """read in review, write to outfile
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/SCARE/, not file!"
    path = str(path)
    id_text_labels = {}
    label_text_feat = []
    outfile_paths = [
            path + "/gliBert_scare_reviews_train.tsv",
            path + "/gliBert_scare_reviews_dev.tsv",
            path + "/gliBert_scare_reviews_test.tsv"
            ]

    rating_text_srl = []

    with open(path + "/scare_v1.0.0_data/reviews/" , "r") as f:
        rows = f.read().split("\n")
    rows_split = [row.split("\t") for row in rows]

    for item in rows_split:
        if item[0] != "":
            application, rating, title, text, date = item
            review = title.rstrip() + " || " + text.lstrip()
            sem_roles = srl_predictor.predict_semRoles(review)
            if sem_roles == False:
                continue
            text_label.append([rating, "", review, sem_roles])

    write_to_files(rating_text_srl, outfile_paths)


def preprocess_XNLI(path):
    """read in merged TSVs, predict SRLs, write label, text and SRLs to new file
    ATTENTION: path points to directory, not input file!
    Args:
        param1: str
        param2: str
        param3: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/XNLI/, not file!"
    path = str(path)
    file_paths = [
            path + "/XNLI-1.0/xnli.dev.de.tsv",
            path + "/XNLI-1.0/xnli.test.de.tsv"
            ]
    outfile_paths = [
            path + "/gliBert_xnli_train.tsv",
            path + "/gliBert_xnli_dev.tsv",
            path + "/gliBert_xnli_test.tsv"
            ]

    for i, file_path in enumerate(file_paths):
        label_text_feat = []
        with open(file_path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            next(f_reader)
            for row in f_reader:
                label, sentence_1, sentence_2 = row[1], row[6], row[7]
                sem_roles_1 = srl_predictor.predict_semRoles(sentence_1)
                sem_roles_2 = srl_predictor.predict_semRoles(sentence_2)
                if sem_roles_1 == False or sem_roles_2 == False:
                    continue
                label_text_feat.append([label, "", sentence_1, sentence_2, sem_roles_1, sem_roles_2])
    
        splitted_write_to_files(label_text_feat, outfile_paths, i)


def preprocess_XQuAD(path):
    """Preprocess XQuAD data
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/XQUAD/, not file!"
    path = str(path)
    outfile_paths = [
            path + "/gliBert_xquad_train.tsv",
            path + "/gliBert_xquad_dev.tsv",
            path + "/gliBert_xquad_test.tsv"
            ]

    too_long_contexts = []
    spans_text_qas_srl = []

    with open(path + "/xquad/xquad.de.json", "r") as f:
        file = f.read()
        json_data = json.loads(file)
    
        for i in range(len(json_data["data"])):
            for j in range(len(json_data["data"][i]["paragraphs"])):
                context = json_data["data"][i]["paragraphs"][j]["context"]
                try:
                    sem_roles_context = srl_predictor.predict_semRoles(context)
                except:
                    too_long_contexts.append(context)
                    continue
                for k in range(len(json_data["data"][i]["paragraphs"][j]["qas"])):
                    question = json_data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                    sem_roles_question = srl_predictor.predict_semRoles(question)
                    if sem_roles_question == False or sem_roles_context == False:
                        continue
                    start_index = json_data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["answer_start"]
                    text = json_data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                    spans_text_qas_srl.append([
                                        start_index,
                                        text, 
                                        context,
                                        question,
                                        sem_roles_context,
                                        sem_roles_question
                                        ])

        write_to_files(spans_text_qas_srl, outfile_paths)

    with open(path + "/too_long.txt", "w") as f:
        for context in too_long_contexts:
            f.write(context + "\n\n")


def main():
    args = parse_cmd_args()
    data_set = args.data_set
    global srl_predictor
    srl_predictor = SRL_predictor(args.argument_model_config, args.bert_path)
    path = args.path
    if data_set == "deISEAR":
        preprocess_deISEAR(path)
    elif data_set == "MLQA":
        preprocess_MLQA(path)
    elif data_set == "PAWS-X":
        preprocess_PAWS_X(path)
    elif data_set == "SCARE":
        preprocess_SCARE(path)
    elif data_set == "XNLI":
        preprocess_XNLI(path)
    elif data_set == "XQuAD":
        preprocess_XQuAD(path)
    

if __name__ == "__main__":
    main()
    
