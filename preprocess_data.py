import csv
import json
import os

from predict_SRL import *

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
        margin = int(((pos -neg)**2)**0.5)
        return ("Positive", True, num) if margin > 1 else ("Positive", False, num)
    else:
        margin = int(((pos -neg)**2)**0.5)
        return ("Negative", True, num) if margin > 1 else ("Negative", False, num)


def preprocess_PAWS_X(path, path_outfile, argument_model_config="../SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"):
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
        

def preprocess_SCARE(path, path_outfile, argument_model_config="../SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"):
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
    argument_model_config = "../SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"
    path_to_data = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/Data/MLQA_V1/dev/dev-context-de-question-de.json"
    path_outfile = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/Data/MLQA_V1/dev/dev-context-de-question-de_srl.json"
    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()
    json_data = read_data(path_to_data)
    #preprocess_MLQA_v1(json_data, dsrl, ParZu_parser, path_outfile)
    

if __name__ == "__main__":
    main()
    
