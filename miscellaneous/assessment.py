import csv
import numpy as np


data_sets = ["deISEAR",
             "SCARE",
             "PAWS-X",
             "XNLI",
             "MLQA",
             "XQuAD"]


def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[1, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)


data_set_counts = {data_set: {"a": 0, "u": 0, "e": 0} for data_set in data_sets}

with open(file) as f:
    rows = [row for row in csv.reader(f)]

for row in rows:
    dataset = re.sub("\d+", "", row[0])
    data_set_counts[dataset][row[1]] += 1
    data_set_counts[dataset][row[2]] += 1
    data_set_counts[dataset][row[3]] += 1

for key, value in data_set_counts.items():
    summe = sum(value.values())
    print(key, {k:round(v/summe*100, 2) for k,v in value.items()})


M = np.asarray([[len([x for x in row[1:] if x == "a"]),
                 len([x for x in row[1:] if x == "u"]),
                 len([x for x in row[1:] if x == "e"])] for row in rows])

