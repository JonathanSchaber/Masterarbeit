import json
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

def preprocess_MLQA_v1(json_data, dsrl, parser):
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
        print("json['data'][{}]['paragraphs'][{}]['context']".format(indices[0], indices[1]))
    import pdb; pdb.set_trace()


def main():
    argument_model_config = "../SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"
    path_to_data = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/Data/MLQA_V1/dev/dev-context-de-question-de.json"
    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()
    json_data = read_data(path_to_data)
    preprocess_MLQA_v1(json_data, dsrl, ParZu_parser)
    

if __name__ == "__main__":
    main()
    
