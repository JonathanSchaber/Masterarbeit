import csv
import torch

#from predict_SRL import *

from torch.utils.data import (
        TensorDataset,
        random_split,
        DataLoader,
        RandomSampler,
        SequentialSampler
        )

from transformers import BertTokenizer


class Dataloader:
    def __init__(self, path_data, path_tokenizer, batch_size):
        self.batch_size = batch_size
        self.path = path_data
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.data = None
        self.max_len = None
        self.y_mapping = None
        self.x_tensor = None
        self.y_tensor = None

######## d e I S E A R ########

class deISEAR_dataloader(Dataloader):
    def load(self):
        """loads the data from deISEAR data set
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
                emotion, sentence = row[1], row[2]
                data.append((emotion, sentence))
                if emotion not in y_mapping:
                    y_mapping[emotion] = counter
                    counter += 1
        self.data = data
        self.y_mapping = y_mapping
    
    def load_torch(self):
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
        self.max_len = longest_sent + 1 if longest_sent < 512 else 512
        print("")
        print("======== Longest sentence ir in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            emotion, sentence = example
            if len(self.tokenizer.tokenize(sentence)) + 1 > 512:
                continue
            x_tensor = self.tokenizer.encode(
                                        sentence,
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation=True, 
                                        return_tensors = 'pt'
                                        )
            x_tensor_list.append(x_tensor)
    #        default_y = torch.tensor([0]*len(self.y_mapping))
    #        default_y[self.y_mapping[emotion]] = 1
    #        y_tensor = default_y
            y_tensor = torch.tensor(self.y_mapping[emotion])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

####################################
########### M L Q A ############

class MLQA_dataloader(Dataloader):
    def load(self):
        """loads the data from MLQA data set
        Args:
            param1: str
        Returns:
            list of tuples of str
            mapping of y
        """
        data = []
        with open(self.path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            for row in f_reader:
                start_span, end_span, context, question = row[0], row[1], row[2], row[3]
                data.append((start_span, end_span, context, question))
    
        self.data = data
    
    def load_torch(self):
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
        longest_sent_1 = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data]) 
        longest_sent_2 = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data]) 
        self.max_len = longest_sent_1 + longest_sent_2 + 1 if longest_sent_1 + longest_sent_2 < 513 else 512
        print("")
        print("======== Longest sentence pair in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            start_span, end_span, context, question = example
            if len(self.tokenizer.tokenize(context)) + len(self.tokenizer.tokenize(question)) + 1 > 512:
                continue
            # Since [CLS] is not part of the sentence, indices must be increased by 1
            start_span = int(start_span) + 1
            end_span = int(end_span) + 1
            x_tensor = self.tokenizer.encode(
                                        context, 
                                        question,
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation=True, 
                                        return_tensors = 'pt'
                                        )
            x_tensor_list.append(x_tensor)
            y_tensor = torch.tensor([start_span, end_span])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

####################################
########### P A W S - X ############

class PAWS_X_dataloader(Dataloader):
    def load(self):
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
                label, sentence_1, sentence_2 = row[0], row[1], row[2]
                data.append((label, sentence_1, sentence_2))
                if label not in y_mapping:
                    y_mapping[label] = counter
                    counter += 1
    
        self.data = data
        self.y_mapping = y_mapping
    
    def load_torch(self):
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
        longest_sent_1 = max([len(self.tokenizer.tokenize(sent[1])) for sent in self.data]) 
        longest_sent_2 = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data]) 
        self.max_len = longest_sent_1 + longest_sent_2 + 1 if longest_sent_1 + longest_sent_2 < 512 else 512
        print("")
        print("======== Longest sentence pair in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            label, sentence_1, sentence_2 = example
            if len(self.tokenizer.tokenize(sentence_1)) + len(self.tokenizer.tokenize(sentence_2)) + 1 > 512:
                continue
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

class SCARE_dataloader(Dataloader):
    def load(self):
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
                label, review = row[0], row[1]
                data.append((label, review))
                if label not in y_mapping:
                    y_mapping[label] = counter
                    counter += 1
    
        self.data = data
        self.y_mapping = y_mapping
    
    def load_torch(self):
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
        self.max_len = longest_sent + 1 if longest_sent < 512 else 512
        print("")
        print("======== Longest sentence in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            label, review = example
            if len(self.tokenizer.tokenize(review)) + 1 > 512:
                continue
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

class XNLI_dataloader(Dataloader):
    def load(self):
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
    
    def load_torch(self):
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
        longest_sent_1 = max([len(self.tokenizer.tokenize(sent[1])) for sent in self.data]) 
        longest_sent_2 = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data]) 
        self.max_len = longest_sent_1 + longest_sent_2 + 1 if longest_sent_1 + longest_sent_2 < 512 else 512
        print("")
        print("======== Longest sentence in data: ========")
        #print("{}".format(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(longest_sent))))
        print("length (tokenized): {}".format(self.max_len))
        for example in self.data:
            label, sentence_1, sentence_2 = example
            if len(self.tokenizer.tokenize(sentence_1)) + len(self.tokenizer.tokenize(sentence_2)) + 1 > 512:
                continue
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
    if data_set == "deISEAR":
        dataloader = deISEAR_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])
    elif data_set == "MLQA":
        dataloader = MLQA_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])
    elif data_set == "PAWS-X":
        dataloader = PAWS_X_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])
    elif data_set == "SCARE":
        dataloader = SCARE_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])
    elif data_set == "XNLI":
        dataloader = XNLI_dataloader(config[location][data_set], config[location]["BERT"], config["batch_size"])

    dataloader.load()
    dataloader.load_torch()
    train_dataloader, test_dataloader = dataloader_torch(
                                            dataloader.x_tensor,
                                            dataloader.y_tensor,
                                            dataloader.batch_size
                                            )
    num_classes = len(dataloader.y_mapping) if dataloader.y_mapping else dataloader.max_len
    mapping = dataloader.y_mapping
    max_len = dataloader.max_len

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


