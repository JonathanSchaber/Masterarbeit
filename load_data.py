import csv

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
        srl_xnli.append((label, sentence1, predict_semRoles(dsrl, process_text(parser, sentence1)), sentence2, predict_semRoles(dsrl, processed(parser, sentence2))))

    return srl_xnli


