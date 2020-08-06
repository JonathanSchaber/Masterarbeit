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

from transformers import BertTokenizer


######## P A W S - X ########

class PAWS_X_dataloader:
    def __init__(self, path_data, path_tokenizer, batch_size):
        self.batch_size = batch_size
        self.path = path_data
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.data = None
        self.max_len = None
        self.y_mapping = None
        self.x_tensor = None
        self.y_tensor = None

    def load_PAWS_X(self):
        """loads the data from PAWS_X data set
        Args:
            param1: str
        Returns:
            list of tuples of str
            mapping of y
        """
        data = []
        y_mapping = {}
        with open(self.path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            counter = 0
            for row in f_reader:
                para_id, sentence_1, sentence_2, label = row[0], row[1]
                data.append((label, sentence_1, sentence_2))
                if label not in y_mapping:
                    y_mapping[label] = counter
                    counter += 1
    
        self.data = data
        self.y_mapping = y_mapping
    
    def load_torch_PAWS_X(self):
        """Return tensor for training
        Args:
            param1: list of tuples of strs
            param2: dict
            param3: torch Tokenizer object
        Returns
            tensor
            tensor
            int
        """
        x_tensor_list = []
        y_tensor_list = []
        longest_sent = max(
                [len(self.tokenizer.tokenize(sent[1])) for sent in self.data] + 
                [len(self.tokenizer.tokenize(sent[2])) for sent in self.data]
                )
        self.max_len = longest_sent + 1 if longest_sent < 200 else 200
        print("")
        print("======== Longest sentence in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            label, sentence_1, sentence_2 = example
            x_tensor = self.tokenizer.encode(
                                        sentence_1, 
                                        sentence_2,
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation=True, 
                                        return_tensors = 'pt'
                                        )
            x_tensor_list.append(x_tensor)
    #        default_y = torch.tensor([0]*len(self.y_mapping))
    #        default_y[self.y_mapping[label]] = 1
    #        y_tensor = default_y
            y_tensor = torch.tensor(self.y_mapping[label])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

#####################################
######## S C A R E ########

class SCARE_dataloader:
    def __init__(self, path_data, path_tokenizer, batch_size):
        self.batch_size = batch_size
        self.path = path_data
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.data = None
        self.max_len = None
        self.y_mapping = None
        self.x_tensor = None
        self.y_tensor = None

    def load_SCARE(self):
        """loads the data from SCARE data set
        Args:
            param1: str
        Returns:
            list of tuples of str
            mapping of y
        """
        data = []
        y_mapping = {}
        with open(self.path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            counter = 0
            for row in f_reader:
                review, label = row[0], row[1]
                data.append((label, review))
                if label not in y_mapping:
                    y_mapping[label] = counter
                    counter += 1
    
        self.data = data
        self.y_mapping = y_mapping
    
    def load_torch_SCARE(self):
        """Return tensor for training
        Args:
            param1: list of tuples of strs
            param2: dict
            param3: torch Tokenizer object
        Returns
            tensor
            tensor
            int
        """
        x_tensor_list = []
        y_tensor_list = []
        longest_sent = max([len(self.tokenizer.tokenize(sent[1])) for sent in self.data])
        self.max_len = longest_sent + 1 if longest_sent < 200 else 200
        print("")
        print("======== Longest sentence in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            label, review = example
            x_tensor = self.tokenizer.encode(
                                        review, 
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation=True, 
                                        return_tensors = 'pt'
                                        )
            x_tensor_list.append(x_tensor)
    #        default_y = torch.tensor([0]*len(self.y_mapping))
    #        default_y[self.y_mapping[label]] = 1
    #        y_tensor = default_y
            y_tensor = torch.tensor(self.y_mapping[label])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

#####################################################################################
######## X N L I #######

class XNLI_dataloader:
    def __init__(self, path_data, path_tokenizer, batch_size):
        self.batch_size = batch_size
        self.path = path_data
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.data = None
        self.max_len = None
        self.y_mapping = None
        self.x_tensor = None
        self.y_tensor = None

    def load_XNLI(self):
        """loads the data from XNLI data set
        Args:
            param1: str
        Returns:
            list of tuples of str
            mapping of y
        """
        data = []
        y_mapping = {}
        with open(self.path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            counter = 0
            for row in f_reader:
                label, sentence_1, sentence_2 = row[1], row[6], row[7]
                data.append((label, sentence_1, sentence_2))
                if label not in y_mapping:
                    y_mapping[label] = counter
                    counter += 1
    
        self.data = data
        self.y_mapping = y_mapping
    
    def load_torch_XNLI(self):
        """Return tensor for training
        Args:
            param1: list of tuples of strs
            param2: dict
            param3: torch Tokenizer object
        Returns
            tensor
            tensor
            int
        """
        x_tensor_list = []
        y_tensor_list = []
        longest_sent = max(
                [len(self.tokenizer.tokenize(sent[1])) for sent in self.data] + 
                [len(self.tokenizer.tokenize(sent[2])) for sent in self.data]
                )
        self.max_len = longest_sent + 1 if longest_sent < 200 else 200
        print("")
        print("======== Longest sentence in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            label, sentence_1, sentence_2 = example
            x_tensor = self.tokenizer.encode(
                                        sentence_1, 
                                        sentence_2, 
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation=True, 
                                        return_tensors = 'pt'
                                        )
            x_tensor_list.append(x_tensor)
    #        default_y = torch.tensor([0]*len(self.y_mapping))
    #        default_y[self.y_mapping[label]] = 1
    #        y_tensor = default_y
            y_tensor = torch.tensor(self.y_mapping[label])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        #import ipdb; ipdb.set_trace()
        #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

#####################################################################################

def dataloader(config, location, data_set):
    """Make XNLI data ready to be passed to transformer dataloader
    Args:
        param1: str
        param2: str
        param3: int
    Returns:
        Dataloader object (train)
        Dataloader object (test)
        int
    """
    if data_set == "XNLI":
        xnli = XNLI_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])
        xnli.load_XNLI()
        xnli.load_torch_XNLI()
        train_dataloader, test_dataloader = dataloader_torch(xnli.x_tensor, xnli.y_tensor, xnli.batch_size)
        num_classes = len(xnli.y_mapping)
        mapping = xnli.y_mapping
        max_len = xnli.max_len
    elif data_set == "SCARE":
        scare = SCARE_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])
        scare.load_SCARE()
        scare.load_torch_SCARE()
        train_dataloader, test_dataloader = dataloader_torch(scare.x_tensor, scare.y_tensor, scare.batch_size)
        num_classes = len(scare.y_mapping)
        mapping = scare.y_mapping
        max_len = scare.max_len

    return train_dataloader, test_dataloader, num_classes, max_len, mapping


def dataloader_torch(x_tensor, y_tensor, batch_size):
    """creates dataloader torch objects
    Args:
        param1: torch tensor
        param2: torch tensor
    Returns:
        torch Dataloader object 
        torch Dataloader object 
    """
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
    return train_dataloader, test_dataloader



#def SRL_XNLI(data, dsrl, parser):
#    """predict semantic roles of xnli data and return new object
#    Args:
#        param1: list of tuples of strs
#        param2: DSRL object
#        param*: ParZu object
#    Returns:
#        list of tuples of strs
#    """
#    srl_xnli = []
#    num_examples = len(data)
#    for i, example in enumerate(data):
#        if i % 100 == 0:
#            print("processed the {}th example out of {}...".format(i, num_examples))
#        label, sentence_1, sentence_2 = example
#        srl_xnli.append((label, sentence_1, sentence_2, predict_semRoles(dsrl, process_text(parser, sentence_1)), predict_semRoles(dsrl, process_text(parser, sentence_2))))
#
#    return srl_xnli


