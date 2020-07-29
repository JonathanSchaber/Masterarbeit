import csv
import torch

#from SemRoleLabeler import *

from torch.utils.data import (
        TensorDataset,
        random_split,
        DataLoader,
        RandomSampler,
        SequentialSampler
        )


def load_XNLI(path):
    """loads the data from XNLI data set
    Args:
        param1: str
    Returns:
        list of tuples of str
        mapping of y
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
        tensor
    """
    x_tensor_list = []
    y_tensor_list = []
    longest_sent = max(max(sent for sent in xnli_data))
    max_len = len(tokenizer.tokenize(longest_sent))
    print("")
    print("======== Longest sentence in data: ========")
    print("{}".format(longest_sent))
    print("length (tokenized): {}".format(max_len))
    for example in xnli_data:
        label, sentence1, sentence2, = example
        #x_tensor = tokenizer.encode(sentence1, sentence2, add_special_tokens = True, truncation=True, return_tensors = 'pt')
        x_tensor = tokenizer.encode(sentence1, sentence2, add_special_tokens = True, max_length = max_len, pad_to_max_length = True, truncation=True, return_tensors = 'pt')
        x_tensor_list.append(x_tensor)
#        default_y = torch.tensor([0]*len(y_mapping))
#        default_y[y_mapping[label]] = 1
#        y_tensor = default_y
        y_tensor = torch.tensor(y_mapping[label])
        y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
    
    #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
    y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 
    x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
    return x_tensor, y_tensor, len(y_mapping)


def dataloader_XNLI(path, tokenizer, batch_size=32):
    """Make XNLI data ready to be passed to transformer dataloader
    Args:
        param1: str
        param2: transformer Tokenizer object
    Returns:
        Dataloader object (train)
        Dataloader object (test)
    """
    data, ys = load_XNLI(path)
    x_tensor, y_tensor, num_classes = load_torch_XNLI(data, ys, tokenizer)

    dataset = TensorDataset(x_tensor, y_tensor)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        ) 
    test_dataloader = DataLoader(
            test_dataset,
            sampler = RandomSampler(test_dataset), 
            batch_size = batch_size 
        ) 
    return train_dataloader, test_dataloader, num_classes


#def SRL_XNLI(xnli_data, dsrl, parser):
#    """predict semantic roles of xnli data and return new object
#    Args:
#        param1: list of tuples of strs
#        param2: DSRL object
#        param*: ParZu object
#    Returns:
#        list of tuples of strs
#    """
#    srl_xnli = []
#    num_examples = len(xnli_data)
#    for i, example in enumerate(xnli_data):
#        if i % 100 == 0:
#            print("processed the {}th example out of {}...".format(i, num_examples))
#        label, sentence1, sentence2 = example
#        srl_xnli.append((label, sentence1, sentence2, predict_semRoles(dsrl, process_text(parser, sentence1)), predict_semRoles(dsrl, process_text(parser, sentence2))))
#
#    return srl_xnli


