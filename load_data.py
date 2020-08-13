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
from pathlib import Path


class Dataloader:
    def __init__(self, path_data, path_tokenizer, batch_size, merge_subtokens):
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.merge_subtokens = merge_subtokens
        self.batch_size = batch_size
        self.path = path_data
        self.path_dev = None
        self.path_test = None
        self.data = None
        self.data_dev = None
        self.data_test = None
        self.max_len = None
        self.attention_mask = None
        self.attention_mask_dev = None
        self.attention_mask_test = None
        self.token_type_ids = None
        self.token_type_ids_dev = None
        self.token_type_ids_test = None
        self.y_mapping = None
        self.x_tensor = None
        self.x_tensor_dev = None
        self.x_tensor_test = None
        self.y_tensor = None
        self.y_tensor_dev = None
        self.y_tensor_test = None
        self.dataset_dev = None
        self.dataset_test = None

    @staticmethod
    def check_max_length(*sent_lengths):
        to_add = 2 if len(sent_lengths) == 1 else 3
        max_length = 0
        for sent_length in sent_lengths:
            max_length += sent_length
        max_length += to_add
        return max_length if max_length < 513 else 512

    @staticmethod
    def merge_subs(subtoken_list): 
        """merges a sub-tokenized sentence back to token level (without special tokens).
        Args:
            param1: list
        Returns:
            list
        """
        token_list = [] 
        for i, token in enumerate(subtoken_list): 
            if token.startswith("##"): 
                continue 
            elif i + 1 == len(subtoken_list): 
                token_list.append(token) 
            elif not subtoken_list[i+1].startswith("##"): 
                token_list.append(token) 
            else: 
                current_word = [token] 
                for subtoken in subtoken_list[i+1:]: 
                    if subtoken.startswith("##"): 
                        current_word.append(subtoken.lstrip("##")) 
                    else: 
                        break 
                token_list.append("".join(current_word)) 
        return token_list  

    def split_dataset(self):
        """if there is no split in original dataset, we create one ourselves
        ratio dev:test = 90:10
        """
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        dev_size = int(0.9 * len(dataset))
        test_size = len(dataset) - dev_size
        self.dataset_dev, self.dataset_test = random_split(dataset, [dev_size, test_size])

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
        with open(str(Path(self.path)) + "/deISEAR_GLIBERT.tsv", "r") as f:
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
        self.max_len = self.check_max_length(longest_sent)
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
            y_tensor = torch.tensor(self.y_mapping[emotion])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

        self.split_dataset()

####################################
########### M L Q A ############

class MLQA_dataloader(Dataloader):
    def load_data(self, data):
        with open(data) as f:
            data = []
            f_reader = csv.reader(f, delimiter="\t")
            for row in f_reader:
                start_index, text, context, question = row[0], row[1], row[2], row[3]
                start_index = int(start_index)
                if not self.merge_subtokens:
                    len_question = len(self.tokenizer.tokenize(question))
                    tokenized_context = self.tokenizer.tokenize(context[:start_index])
                    start_span = len(tokenized_context)
                    end_span = start_span + len(self.tokenizer.tokenize(text)) - 1
                    start_span += len_question + 1
                    end_span += len_question + 1
                else:
                    len_question = len(self.merge_subs(self.tokenizer.tokenize(question)))
                    tokenized_context = self.tokenizer.tokenize(context[:start_index])
                    start_span = len(self.merge_subs(tokenized_context))
                    end_span = start_span + len(self.merge_subs(self.tokenizer.tokenize(text))) - 1
                    start_span += len_question + 1
                    end_span += len_question + 1

                data.append((start_span, end_span, context, question))
        return data

    def get_max_len(self):
        longest_context_dev = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data_dev]) 
        longest_question_dev = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_dev]) 
        longest_context_test = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data_test]) 
        longest_question_test = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_test]) 
        longest_context = max(longest_context_dev, longest_context_test)
        longest_question = max(longest_question_dev, longest_question_test)
        return self.check_max_length(longest_context, longest_question)

    def load_torch_data(self, data):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        y_tensor_list = []

        self.max_len = self.get_max_len()
        for example in data:
            start_span, end_span, context, question = example
            if len(self.tokenizer.tokenize(context)) + len(self.tokenizer.tokenize(question)) + 1 > 512:
                continue
            # Since [CLS] is not part of the sentence, indices must be increased by 1
            start_span = int(start_span) + 1
            end_span = int(end_span) + 1
            encoded_dict = self.tokenizer.encode_plus(
                                        question, 
                                        context,
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation = True, 
                                        return_tensors = 'pt',
                                        return_token_type_ids = True,
                                        return_attention_mask = True
                                        )
            input_ids.append(encoded_dict["input_ids"])
            attention_mask.append(encoded_dict["attention_mask"])
            token_type_ids.append(encoded_dict["token_type_ids"])
            y_tensor = torch.tensor([start_span, end_span])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        return torch.cat(input_ids, dim=0), \
                torch.cat(attention_mask, dim=0), \
                torch.cat(token_type_ids, dim=0), \
                torch.cat(tuple(y_tensor_list), dim=0) 

    def load(self):
        self.path_dev = str(Path(self.path)) + "/dev/dev-context-de-question-de.tsv"
        self.path_test = str(Path(self.path)) + "/test/test-context-de-question-de.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)
    
    def load_torch(self):
        self.x_tensor_dev, \
        self.attention_mask_dev, \
        self.token_type_ids_dev, \
        self.y_tensor_dev = self.load_torch_data(self.data_dev)

        self.x_tensor_test, \
        self.attention_mask_test, \
        self.token_type_ids_test, \
        self.y_tensor_test = self.load_torch_data(self.data_test)

        print("")
        print("======== Longest sentence pair in data: ========")
        print("length (tokenized): {}".format(self.max_len))

        self.dataset_dev = TensorDataset(
                                self.x_tensor_dev,
                                self.y_tensor_dev,
                                self.attention_mask_dev,
                                self.token_type_ids_dev
                                )
        self.dataset_test = TensorDataset(
                                self.x_tensor_test,
                                self.y_tensor_test,
                                self.attention_mask_test,
                                self.token_type_ids_test
                                )

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
        self.max_len = self.check_max_length(longest_sent_1, longest_sent_2)
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
        self.max_len = self.check_max_length(longest_sent)
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
        self.max_len = self.check_max_length(longest_sent_1, longest_sent_2)
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
            y_tensor = torch.tensor(self.y_mapping[label])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        #y_tensor = torch.unsqueeze(torch.tensor(y_tensor_list), dim=1)
        self.x_tensor = torch.cat(tuple(x_tensor_list), dim=0) 
        self.y_tensor = torch.cat(tuple(y_tensor_list), dim=0) 

####################################
########### X Q u A D ############

class XQuAD_dataloader(Dataloader):
    def __init__(self, path_data, path_tokenizer, batch_size, merge_subtokens):
        self.batch_size = batch_size
        self.merge_subtokens = merge_subtokens
        self.path = path_data
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.data = None
        self.max_len = None
        self.y_mapping = None
        self.x_tensor = None
        self.attention_mask = None
        self.token_type_ids = None
        self.y_tensor = None

    @staticmethod
    def merge_subs(subtoken_list): 
        """merges a sub-tokenized sentence back to token level (without special tokens).
        Args:
            param1: list
        Returns:
            list
        """
        token_list = [] 
        for i, token in enumerate(subtoken_list): 
            if token.startswith("##"): 
                continue 
            elif i + 1 == len(subtoken_list): 
                token_list.append(token) 
            elif not subtoken_list[i+1].startswith("##"): 
                token_list.append(token) 
            else: 
                current_word = [token] 
                for subtoken in subtoken_list[i+1:]: 
                    if subtoken.startswith("##"): 
                        current_word.append(subtoken.lstrip("##")) 
                    else: 
                        break 
                token_list.append("".join(current_word)) 
        return token_list  

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
                start_index, text, context, question = row[0], row[1], row[2], row[3]
                start_index = int(start_index)
                if not self.merge_subtokens:
                    len_question = len(self.tokenizer.tokenize(question))
                    tokenized_context = self.tokenizer.tokenize(context[:start_index])
                    start_span = len(tokenized_context)
                    end_span = start_span + len(self.tokenizer.tokenize(text)) - 1
                    start_span += len_question + 1
                    end_span += len_question + 1
                else:
                    len_question = len(self.merge_subs(self.tokenizer.tokenize(question)))
                    tokenized_context = self.tokenizer.tokenize(context[:start_index])
                    start_span = len(self.merge_subs(tokenized_context))
                    end_span = start_span + len(self.merge_subs(self.tokenizer.tokenize(text))) - 1
                    start_span += len_question + 1
                    end_span += len_question + 1

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
        input_ids = []
        attention_mask = []
        token_type_ids = []
        y_tensor_list = []

        longest_sent_1 = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data]) 
        longest_sent_2 = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data]) 
        self.max_len = self.check_max_length(longest_sent_1, longest_sent_2)
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
            encoded_dict = self.tokenizer.encode_plus(
                                        question, 
                                        context,
                                        add_special_tokens = True, 
                                        max_length = self.max_len,
                                        pad_to_max_length = True, 
                                        truncation = True, 
                                        return_tensors = 'pt',
                                        return_token_type_ids = True,
                                        return_attention_mask = True
                                        )
            input_ids.append(encoded_dict["input_ids"])
            attention_mask.append(encoded_dict["attention_mask"])
            token_type_ids.append(encoded_dict["token_type_ids"])
            y_tensor = torch.tensor([start_span, end_span])
            y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        
        self.x_tensor = torch.cat(input_ids, dim=0)
        self.attention_mask = torch.cat(attention_mask, dim=0)
        self.token_type_ids = torch.cat(token_type_ids, dim=0)
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
        dataloader = deISEAR_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"]
                            )
    elif data_set == "MLQA":
        dataloader = MLQA_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"]
                            )
    elif data_set == "PAWS-X":
        dataloader = PAWS_X_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"]
                            )
    elif data_set == "SCARE":
        dataloader = SCARE_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"]
                            )
    elif data_set == "XNLI":
        dataloader = XNLI_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"]
                            )
    elif data_set == "XQuAD":
        dataloader = XQuAD_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"]
                            )

    dataloader.load()
    dataloader.load_torch()
    dev_dataloader = DataLoader(
            dataloader.dataset_dev,
            sampler = RandomSampler(dataloader.dataset_dev),
            batch_size = dataloader.batch_size
        ) 
    test_dataloader = DataLoader(
            dataloader.dataset_test,
            sampler = RandomSampler(dataloader.dataset_test),
            batch_size = dataloader.batch_size
        ) 
#    train_dataloader, test_dataloader = dataloader_torch(
#                                            dataloader.x_tensor,
#                                            dataloader.y_tensor,
#                                            attention_mask=dataloader.attention_mask,
#                                            token_type_ids=dataloader.token_type_ids,
#                                            batch_size=dataloader.batch_size
#                                            )
    num_classes = len(dataloader.y_mapping) if dataloader.y_mapping else dataloader.max_len
    mapping = dataloader.y_mapping
    max_len = dataloader.max_len

    return dev_dataloader, test_dataloader, num_classes, max_len, mapping


#def dataloader_torch(x_tensor, y_tensor, attention_mask=None, token_type_ids=None, batch_size=None):
#    """creates dataloader torch objects
#    Args:
#        param1: torch tensor
#        param2: torch tensor
#    Returns:
#        torch Dataloader object 
#        torch Dataloader object 
#    """
#    if token_type_ids == None:
#        dataset = TensorDataset(x_tensor, y_tensor)
#    else:
#        dataset = TensorDataset(x_tensor, y_tensor, attention_mask, token_type_ids)
#    train_size = int(0.9 * len(dataset))
#    test_size = len(dataset) - train_size
#    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#    train_dataloader = DataLoader(
#            train_dataset,
#            sampler = RandomSampler(train_dataset),
#            batch_size = batch_size
#        ) 
#    test_dataloader = DataLoader(
#            test_dataset,
#            sampler = RandomSampler(test_dataset), 
#            batch_size = batch_size 
#        ) 
#    return train_dataloader, test_dataloader
#
