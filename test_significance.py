import random

from typing import List


def permute_without_replacement(treatment: List[bool], control: List[bool]) -> bool:
    """compute random permutations, t(X, Y)

    Args:
        treatment: True, False ordered list
        control: True, False ordered list
    Returns:
        True if t(X, Y), else False
    """
    r, R = 0, 0

    e_treat_orig = len([x for x in treatment if x])/len(treatment)
    e_contr_orig = len([x for x in control if x])/len(control)

    t_orig = e_treat_orig - e_contr_orig

    while R < 1000:

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
            R += 1

        treatment = perm_treat
        control = perm_contr

    print(r/R)


def permute_with_replacement(treatment: List[bool], control: List[bool]) -> bool:
    """compute random permutations, t(X, Y)

    Args:
        treatment: True, False ordered list
        control: True, False ordered list
    Returns:
        True if t(X, Y), else False
    """
    r, R = 0, 0

    e_treat_orig = len([x for x in treatment if x])/len(treatment)
    e_contr_orig = len([x for x in control if x])/len(control)

    t_orig = e_treat_orig - e_contr_orig

    while R < 1000:

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
            R += 1

    print(r/R)

