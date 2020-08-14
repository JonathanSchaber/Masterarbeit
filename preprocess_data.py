import argparse
import csv
import json
import os

from random import shuffle
from pathlib import Path
from predict_SRL import SRL_predictor


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
            help="Path to file ATTENTION in case of PAWS-X this points to the directory containing the files, not the files itself!",
            )
    return parser.parse_args()


def get_majority_label(labels):
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
    ATTENTION: path to root directory, not file(s)!
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/deISEAR/, not file!"
    path = str(path)
    file_path = path + "/deISEAR.tsv"
    outfile_paths = [path + "/GLIBERT_deISEAR_dev.tsv", path + "/GLIBERT_deISEAR_test.tsv"]

    emotion_sentence_srl = []

    with open(file_path, "r") as f:
        f_reader = csv.reader(f, delimiter="\t")
        next(f_reader)
        for i, row in enumerate(f_reader):
            emotion, sentence = row[1], row[2]
            sem_roles = srl_predictor.predict_semRoles(sentence)
            if i % 50 == 0 and i != 0:
                print("")
                print("Senstence: {}".format(sentence))
                for sentence in sem_roles:
                    for sem_roles in sentence:
                        print("Predicted SRLs: {}".format(element))
            emotion_sentence_srl.append([emotion, "", sentence, sem_roles])
    
    len_dev = int(len(emotion_sentence_srl)*0.9)
    len_test = len(emotion_sentence_srl) - len_dev
    shuffle(emotion_sentence_srl)

    with open(outfile_paths[0], "w") as f:
        for element in emotion_sentence_srl[:len_dev]:
            csv.writer(f, delimiter="\t").writerow(element)
    with open(outfile_paths[1], "w") as f:
        for element in emotion_sentence_srl[-len_test:]:
            csv.writer(f, delimiter="\t").writerow(element)


def preprocess_MLQA(path):
    """Preprocess MLQA data
    ATTENTION: path to root directory, not file(s)!
    Args:
        param1: str
        param2: str
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
            path + "/dev/GLIBERT_dev-context-de-question-de.tsv",
            path + "/test/GLIBERT_test-context-de-question-de.tsv"
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
                    if i % 50 == 0:
                        print("")
                        print("Context: {}".format(context))
                        for sentence in sem_roles_context:
                            for sem_roles in sentence:
                                print("Predicted SRLs: {}".format(sem_roles))
                        print("Question: {}".format(question))
                        for sentence in sem_roles_question:
                            for sem_roles in sentence:
                                print("Predicted SRLs: {}".format(sem_roles))
                    spans_text_qas_srl.append([
                                            start_index,
                                            text, context,
                                            question,
                                            sem_roles_context,
                                            sem_roles_question
                                            ])

        with open(outfile_paths[h], "w") as f:
            for element in spans_text_qas_srl:
                csv.writer(f, delimiter="\t").writerow(element)
    with open("too_long.txt", "w") as f:
        for context in too_long_contexts:
            f.write(context + "\n")


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
    file_paths = [path + "/de/dev_2k.tsv", path + "/de/test_2k.tsv"]
    outfile_paths = [path + "/de/GLIBERT_paws_x_dev.tsv", path + "/de/GLIBERT_paws_x_test.tsv"]

    label_text_feat = []

    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            next(f_reader)
            for row in f_reader:
                para_id, sentence_1, sentence_2, label = row[0], row[1], row[2], row[3]
                sem_roles_1 = srl_predictor.predict_semRoles(sentence_1)
                sem_roles_2 = srl_predictor.predict_semRoles(sentence_2)
                label_text_feat.append([label, "", sentence_1, sentence_2, sem_roles_1, sem_roles_2])
    
        with open(outfile_paths[i], "w") as f:
            for element in label_text_feat:
                csv.writer(f, delimiter="\t").writerow(element)


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
            path + "/scare_v1.0.0/annotations/GLIBERT_annotations_dev.tsv",
            path + "/scare_v1.0.0/annotations/GLIBERT_annotations_test.tsv"
            ]

    count_non_maj = 0
    count_no_labels = 0
    count_close = 0
    count_all = 0

    with open(path + "/scare_v1.0.0/annotations/annotations.txt", "r") as f:
        ids_texts = [example.split("\t") for example in f.read().split("\n")]
    with open(path + "/scare_v1.0.0/annotations/annotations.csv", "r") as f:
        rows = f.read().split("\n")
        ids_labels = []
        for row in rows:
            if row != "":
                entity, review_id, left, right, string, phrase_id, polarity, relation = row.split("\t")
                ids_labels.append([review_id, polarity])

    ids_texts.pop()
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
        label_text_feat.append([polarity, feat["text"], sem_roles])
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
    print("")
    print("======== Writing to file: {} ========".format(path_outfile))

    len_dev = int(len(label_text_feat)*0.9)
    len_test = len(label_text_feat) - len_dev
    shuffle(label_text_feat)

    with open(outfile_path[0], "w") as f:
        for element in label_text_feat[:len_dev]:
            csv.writer(f, delimiter="\t").writerow(element)
    with open(outfile_path[1], "w") as f:
        for element in label_text_feat[-len_test:]:
            csv.writer(f, delimiter="\t").writerow(element)


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
            path + "/scare_v1.0.0_data/GLIBERT_annotations_dev.tsv",
            path + "/scare_v1.0.0_data/GLIBERT_annotations_test.tsv"
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
            text_label.append([rating, "", review, sem_roles])

    with open(path_outfile, "w") as f:
        for element in text_label:
            csv.writer(f, delimiter="\t").writerow(element)


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
    file_paths = [path + "/XNLI-1.0/xnli.dev.de.tsv", path + "/XNLI-1.0/xnli.test.de.tsv"]
    outfile_paths = [path + "/XNLI-1.0/GLIBERT_xnli.dev.de.tsv", path + "/XNLI-1.0/GLIBERT_xnli.test.de.tsv"]

    label_text_feat = []

    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            next(f_reader)
            for row in f_reader:
                sentence_1, sentence_2, label = row[1], row[6], row[7]
                sem_roles_1 = srl_predictor.predict_semRoles(sentence_1)
                sem_roles_2 = srl_predictor.predict_semRoles(sentence_2)
                label_text_feat.append([label, sentence_1, sentence_2, sem_roles_1, sem_roles_2])
    
        with open(outfile_paths[i], "w") as f:
            for element in label_text_feat:
                csv.writer(f, delimiter="\t").writerow(element)


def preprocess_XQuAD(path):
    """Preprocess XQuAD data
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    path = Path(path)
    assert path.is_dir(), "Path must point to root directory /<path>/<to>/MLQA/, not file!"
    spans_text_qas_srl = []
    path_outfile = str(Path(path).parent) + "/XQuAD.tsv"

    with open(path, "r") as f:
        file = f.read()
        json_data = json.loads(file)
    
    for i in range(len(json_data["data"])):
        for j in range(len(json_data["data"][i]["paragraphs"])):
            context = json_data["data"][i]["paragraphs"][j]["context"]
            for k in range(len(json_data["data"][i]["paragraphs"][j]["qas"])):
                question = json_data["data"][i]["paragraphs"][j]["qas"][k]["question"]
                start_index = json_data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["answer_start"]
                text = json_data["data"][i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]
                spans_text_qas_srl.append([start_index, text, context, question])

    with open(path_outfile, "w") as f:
        for element in spans_text_qas_srl:
            csv.writer(f, delimiter="\t").writerow(element)


def main():
    args = parse_cmd_args()
    data_set = args.data_set
    global srl_predictor
    srl_predictor = SRL_predictor(args.argument_model_config)
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
    
