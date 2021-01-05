import argparse
import random

from ensemble_predict import build_dicts, get_true_best_dev_epoch, read_in_files
from typing import List


def parse_cmd_args():
    """Parse command line arguments.

    Returns:
        parser: argparse object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-t",
            "--treatment_files",
            type=str,
            nargs="+",
            help="List of treatment files for ensemble prediciton"
            )
    parser.add_argument(
            "-c",
            "--control_files",
            type=str,
            nargs="+",
            help="List of control files for ensemble prediciton"
            )
    parser.add_argument(
            "-d",
            "--delete_result_files",
            action="store_true",
            help="Clean up results.json-files"
            )
    parser.add_argument(
            "-R",
            "--R",
            type=int,
            default=1000,
            help="R value: number of permutations"
            )
    return parser.parse_args()


def permute_without_replacement(treatment: List[bool], control: List[bool], R: int=1000) -> bool:
    """compute random permutations, t(X, Y)

    Args:
        treatment: True, False ordered list
        control: True, False ordered list
        R: number of permutations
    count_Returns:
        None
    """
    r, count_R = 0, 0

    e_treat_orig = len([x for x in treatment if x])/len(treatment)
    e_contr_orig = len([x for x in control if x])/len(control)

    t_orig = e_treat_orig - e_contr_orig

    while count_R < R+1:

        perm_treat, perm_contr = [], []

        for item in zip(treatment, control):
            if random.randint(0,100) > 50:
                perm_treat.append(item[0])
                perm_contr.append(item[1])
            else:
                perm_treat.append(item[1])
                perm_contr.append(item[0])

        e_treat = len([x for x in perm_treat if x])/len(perm_treat)
        e_contr = len([x for x in perm_contr if x])/len(perm_contr)

        t_perm = e_treat - e_contr

        if t_perm >= t_orig:
            r += 1
        else:
            count_R += 1

        treatment = perm_treat
        control = perm_contr

    print("r:", r)
    print("R:", R)
    print("(r+1)/(R+1):", round((r+1)/(R+1), 4))


def permute_with_replacement(treatment: List[bool], control: List[bool], R: int=1000) -> bool:
    """compute random permutations, t(X, Y)

    Args:
        treatment: True, False ordered list
        control: True, False ordered list
        R: number of permutations
    Returns:
        None
    """
    r, count_R = 0, 0

    e_treat_orig = len([x for x in treatment if x])/len(treatment)
    e_contr_orig = len([x for x in control if x])/len(control)

    t_orig = e_treat_orig - e_contr_orig

    while count_R < R+1:

        perm_treat, perm_contr = [], []

        for item in zip(treatment, control):
            if random.randint(0,100) > 50:
                perm_treat.append(item[0])
                perm_contr.append(item[1])
            else:
                perm_treat.append(item[1])
                perm_contr.append(item[0])

        e_treat = len([x for x in perm_treat if x])/len(perm_treat)
        e_contr = len([x for x in perm_contr if x])/len(perm_contr)

        t_perm = e_treat - e_contr

        if t_perm >= t_orig:
            r += 1
        else:
            count_R += 1

    print("r:", r)
    print("R:", R)
    print("(r+1)/(R+1):", round((r+1)/(R+1), 4))

def main():
    args = parse_cmd_args()
    treatment_files = list(dict.fromkeys(args.treatment_files))
    control_files = list(dict.fromkeys(args.control_files))
    #print(files)
    treatment_jsons = read_in_files(treatment_files, args.delete_result_files)
    control_jsons = read_in_files(control_files, args.delete_result_files)

    treatment, control = [], []
    treatment_tf, control_tf = [], []
    treatment_dict, control_dict, gold_dict, gold_dict2 = {}, {}, {}, {}
    for file_pair in treatment_jsons:
            _, results_file = file_pair
            true_best_epoch, true_best_dev = get_true_best_dev_epoch(results_file, qa_flag=False)
            test = results_file[true_best_epoch][str(true_best_epoch)]["test"]
            treatment.append(test)
    for file_pair in control_jsons:
            _, results_file = file_pair
            true_best_epoch, true_best_dev = get_true_best_dev_epoch(results_file, qa_flag=False)
            test = results_file[true_best_epoch][str(true_best_epoch)]["test"]
            control.append(test)
    for results in treatment:
        build_dicts(results, treatment_dict, gold_dict)
    for results in control:
        build_dicts(results, control_dict, gold_dict2)

    treatment_ensemble = {key: max(value, key=value.count) for key, value in treatment_dict.items()}
    control_ensemble = {key: max(value, key=value.count) for key, value in control_dict.items()}

    for i in sorted(gold_dict.items()):
        if treatment_ensemble[i[0]] == i[1]:
            treatment_tf.append(1)
        else:
            treatment_tf.append(0)
        if control_ensemble[i[0]] == i[1]:
            control_tf.append(1)
        else:
            control_tf.append(0)

    print("")
    print("Permute with replacement:")
    permute_with_replacement(treatment_tf, control_tf, args.R)
    print("")
    print("Permute without replacement:")
    permute_without_replacement(treatment_tf, control_tf, args.R)



if __name__ == "__main__":
    main()
