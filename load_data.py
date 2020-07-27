import csv
import torch

from SemRoleLabeler import *

def load_XNLI(path):
    """loads the data from XNLI data set
    Args:
        param1: str
    Returns:
        list of tuples of str, mapping of y
    """
    xnli_data = []
    y_mapping = {}
    with open(path, "r") as f:
        f_reader = csv.reader(f, delimiter="\t")
        counter = 0
        for row in f_reader:
            label, sentence1, sentence2 = row[1], row[6], row[7]
            xnli_data.append((label, sentence1, sentence2))
            if label not in y_mapping:
                y_mapping[label] = counter
                counter += 1

    return xnli_data, y_mapping


def load_torch_XNLI(xnli_data, y_mapping, tokenizer):
    """Return tensor for training
    Args:
        param1: list of tuples of strs
        param2: dict
    Returns
        tensor
    """
    max_len = 200
    x_tensor_list = []
    y_tensor_list = []
    for example in xnli_data:
        label, sentence1, sentence2, = example
        x_tensor_list.append(tokenizer.encode(sentence1, sentence2, add_special_tokens = True, max_length = max_len, pad_to_max_length = True, return_tensors = 'pt'))
        y_tensor_list.append(y_mapping[label])
    
    y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
    x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
    import pdb; pdb.set_trace()
    return torch.cat(tuple(x_tensor_list), dim=1), torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)


def SRL_XNLI(xnli_data, dsrl, parser):
    """predict semantic roles of xnli data and return new object
    Args:
        param1: list of tuples of strs
        param2: DSRL object
        param*: ParZu object
    Returns:
        list of tuples of strs
    """
    srl_xnli = []
    num_examples = len(xnli_data)
    for i, example in enumerate(xnli_data):
        if i % 100 == 0:
            print("processed the {}th example out of {}...".format(i, num_examples))
        label, sentence1, sentence2 = example
        srl_xnli.append((label, sentence1, sentence2, predict_semRoles(dsrl, process_text(parser, sentence1)), predict_semRoles(dsrl, process_text(parser, sentence2))))

    return srl_xnli


