import csv

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

