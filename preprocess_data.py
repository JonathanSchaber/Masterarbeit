import csv
import json
import os

from pathlib import Path
from predict_SRL import *

argument_model_config = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"


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
            help="Indicate on which data set model should be trained",
            choices=["XNLI", "SCARE", "PAWS-X"]
            )
    parser.add_argument(
            "-p", 
            "--path", 
            type=str, 
            help="Path to file ATTENTION in case of PAWS-X this points to the directory containing the files, not the files itself!",
            )
    return parser.parse_args()


def read_data(path):
    """reads JSON from file
    Args:
        param1: str
    Returns:
        JSON object
    """
    with open(path, "r") as f:
        file = f.read()

    json_data = json.loads(file)
    return json_data


def SRL_MLQA_v1(json_data, dsrl, parser, path_outfile):
    """processes json data, predicts sem_roles, writes to new file
    Args:
        param1: json data
        param2: DAMESRL object
        param3: ParZu parser object
    Returns:
        None
    """
    failed_texts = []
    for i in range(len(json_data["data"])):
        if i % 20 == 0:
            print("Processed the {}th element...".format(i))
        for j in range(len(json_data["data"][i]["paragraphs"])):
            try:
                srl_context = predict_semRoles(dsrl, process_text(parser, json_data["data"][i]["paragraphs"][j]["context"]))
                json_data["data"][i]["paragraphs"][j]["srl_context"] = srl_context
            except:
                print(json_data["data"][i]["paragraphs"][j]["context"])
                failed_texts.append((i, j))
    print("The following texts were not processed\n:")
    for indices in failed_texts:
        print("json_data['data'][{}]['paragraphs'][{}]['context']".format(indices[0], indices[1]))

    write_obj = json.dumps(json_data)
    try:
        with open(path_outfile, "w", encoding="utf8") as f:
            f.write(write_obj)
    except:
        import pdb; pdb.set_trace()


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


def preprocess_MLQA(path, argument_model_config):
    """Preprocess MLQA data
    ATTENTION: path to root directory, not file(s)!
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    spans_text_qas_srl = []
    path_outfile = str(Path(path).parent) + "/paws-x_SRL.tsv"

    file_paths = [path + "dev/dev-context-de-question-de.json", path + "test/test-context-de-question-de.json"]

    for path in file_paths:
        with open(path, "r") as f:
            file = f.read()
            json_data = json.loads(file)
        
        for i in range(len(json_data["data"])):
            for j in range(len(json_data["data"][i]["paragraphs"])):
                text = json["data"][i]["paragrahps"][i]["context"]
                for k in json_data["data"][i]["paragraphs"][i]["qas"]:
                    question = json_data["data"][i]["paragraphs"][i]["qas"]["question"]
                    start_span = json_data["data"][i]["paragraphs"][i]["qas"]["answer_start"]
                    spans_text_qas_srl.append(start_span, text, question)

    with open(path_outfile, "w"):
        for element in spans_text_qas_srl:
            csv.writer(f, delimiter="\t").writerow(element)



def preprocess_PAWS_X(path, argument_model_config):
    """read in merged TSVs, predict SRLs, write label, text and SRLs to new file
    ATTENTION: path points to directory, not input file!
    Args:
        param1: str
        param2: str
        param3: str
    Returns:
        None
    """
    label_text_feat = []
    path_outfile = str(Path(path).parent) + "/paws-x_SRL.tsv"

    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()

    with open(path, "r") as f:
        f_reader = csv.reader(f, delimiter="\t")
        for row in f_reader:
            para_id, sentence_1, sentence_2, label = row[0], row[1], row[2], row[3]
            dsrl_obj_1 = process_text(ParZu_parser, sentence_1)
            sem_roles_1 = predict_semRoles(dsrl, dsrl_obj_1)
            dsrl_obj_2 = process_text(ParZu_parser, sentence_2)
            sem_roles_2 = predict_semRoles(dsrl, dsrl_obj_2)
            label_text_feat.append([label, sentence_1, sentence_2, sem_roles_1, sem_roles_2])

    with open(path_outfile, "w") as f:
        for element in label_text_feat:
            csv.writer(f, delimiter="\t").writerow(element)
        

def preprocess_SCARE(path, argument_model_config):
    """read in merged TSVs, predict SRLs, write label, text and SRLs to new file
    ATTENTION: path points to directory, not input file!
    Args:
        param1: str
        param2: str
        param3: str
    Returns:
        None
    """
    id_text_labels = {}
    label_text_feat = []
    path_outfile = str(Path(path).parent) + "/annotations_SRL.tsv"

    count_non_maj = 0
    count_no_labels = 0
    count_close = 0
    count_all = 0

    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()

    with open(path + "annotations.txt", "r") as f:
        ids_texts = [example.split("\t") for example in f.read().split("\n")]
    with open(path + "annotations.csv", "r") as f:
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
        dsrl_obj = process_text(ParZu_parser, feat["text"])
        sem_roles = predict_semRoles(dsrl, dsrl_obj)
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

    with open(path_outfile, "w") as f:
        for element in label_text_feat:
            csv.writer(f, delimiter="\t").writerow(element)


def preprocess_SCARE_reviews(path, path_outfile):
    """read in review, write to outfile
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    text_label = []

    with open(path, "r") as f:
        rows = f.read().split("\n")
    rows_split = [row.split("\t") for row in rows]

    for item in rows_split:
        if item[0] != "":
            application, rating, title, text, date = item
            text_label.append([title.rstrip() + " || " + text.lstrip(), rating])

    with open(path_outfile, "w") as f:
        for element in text_label:
            csv.writer(f, delimiter="\t").writerow(element)


def main():
    args = parse_cmd_args()
    if args.argument_model_config:
        global argument_model_config
        argument_model_config = args.argument_model_config 
    data_set = args.data_set
    path = args.path
    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()
    #json_data = read_data(path_to_data)
    if data_set == "PAWS-X":
        preprocess_SCARE(path, argument_model_config)
    elif data_set == "SCARE":
        preprocess_PAWS_X(path, argument_model_config)
    elif data_set == "XNLI":
        preprocess_PAWS_X(path, argument_model_config)
    elif data_set == "XQuAD":
        preprocess_PAWS_X(path, argument_model_config)
    

if __name__ == "__main__":
    main()
    
