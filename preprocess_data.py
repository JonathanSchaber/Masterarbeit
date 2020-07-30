import json
import os

from SemRoleLabeler import *

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


def SRL_MLQA_v1(json_data, dsrl, parser, path_to_new_file):
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
        with open(path_to_new_file, "w", encoding="utf8") as f:
            f.write(write_obj)
    except:
        import pdb; pdb.set_trace()


def get_majority_label(sentiments, labels):
    pass


def preprocess_SCARE(path, path_to_new_file):
    """merge all TSV files, write text and label to new file
    ATTENTION: path points to directory, not files!
    Args:
        param1: str
        param2: str
    Returns:
        None
    """
    sentiments = ["Positive", "Negative", "Neutral"]
    id_text_labels = {}
    text_label = []

    count_non_maj = 0
    count_close = 0
    count_all = 0

    with open(path + "annotations.txt", "r") as f:
        ids_texts = [example.split("\t") for example in f.read().split("\n")]
    with open(path + "annotations.csv", "r") as f:
        ids_labels = []
        for row in csv.reader(f):
            try:
                entity, review_id, left, right, string, phrase_id, polarity, relation = row[0].split("\t")
                ids_labels.append([review_id, polarity])
            except IndexError:
                continue
            except ValueError:
                continue

    ids_texts.pop()
    for id, text in ids_texts:
        if id in id_text_labels:
            raise ERRRRRRORRRRRRR
        else:
            id_text_labels[id] = {"text": text, "labels": []}

    for id, label in ids_labels:
        if id not in id_text_labels:
            raise ERRRRRRRRRRRRRRRROR
        else:
            id_text_labels[id]["labels"].append(label)
    #TODO: in .csv get all polarities for id, then majority decision what whole text gets labeled
    return id_text_labels


def main():
    argument_model_config = "../SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"
    path_to_data = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/Data/MLQA_V1/dev/dev-context-de-question-de.json"
    path_to_outfile = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/Data/MLQA_V1/dev/dev-context-de-question-de_srl.json"
    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()
    json_data = read_data(path_to_data)
    preprocess_MLQA_v1(json_data, dsrl, ParZu_parser, path_to_outfile)
    

if __name__ == "__main__":
    main()
    
