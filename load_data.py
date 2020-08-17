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
        self.data_dev = None
        self.data_test = None
        self.max_len = None
        self.type = None
        self.attention_mask_dev = None
        self.attention_mask_test = None
        self.token_type_ids_dev = None
        self.token_type_ids_test = None
        self.y_mapping = {}
        self.x_tensor_dev = None
        self.x_tensor_test = None
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

    def load_data(self, path):
        data = []
        y_mapping = self.y_mapping
        with open(path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            counter = 0
            for row in f_reader:
                if self.type == 1:
                    label, blank, sentence = row[0], row[1], row[2]
                    data.append((label, blank, sentence, ""))
                elif self.type == 2:
                    label, blank, sentence_1, sentence_2 = row[0], row[1], row[2], row[3]
                    data.append((label, blank, sentence_1, sentence_2))
                elif self.type == "qa":
                    start_index, text, context, question, = row[0], row[1], row[2], row[3]
                    start_index = int(start_index)
                    tokenized_context = self.tokenizer.tokenize(context[:start_index])
                    if not self.merge_subtokens:
                        len_question = len(self.tokenizer.tokenize(question))
                        start_span = len(tokenized_context)
                        end_span = start_span + len(self.tokenizer.tokenize(text)) - 1
                    else:
                        len_question = len(self.merge_subs(self.tokenizer.tokenize(question)))
                        start_span = len(self.merge_subs(tokenized_context))
                        end_span = start_span + len(self.merge_subs(self.tokenizer.tokenize(text))) - 1
                    start_span += len_question + 1
                    end_span += len_question + 1
                    data.append((start_span, end_span, question, context))

                if self.type != "qa":
                    if label not in y_mapping:
                        y_mapping[label] = counter
                        counter += 1
    
        self.y_mapping = y_mapping
        return data

    def make_datasets(self):
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

    def get_max_len(self):
        if type(self.data_dev[3]) != list:
            longest_sentence_1_dev = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data_dev]) 
            longest_sentence_2_dev = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_dev]) 
            longest_sentence_1_test = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data_test]) 
            longest_sentence_2_test = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_test]) 
            longest_sentence_1 = max(longest_sentence_1_dev, longest_sentence_1_test)
            longest_sentence_2 = max(longest_sentence_2_dev, longest_sentence_2_test)
            return self.check_max_length(longest_sentence_1, longest_sentence_2)
        else:
            longest_sent = max([len(self.tokenizer.tokenize(sent[2])) for sent in self.data]) 
            return self.check_max_length(longest_sent)

    def load_torch_data(self, data):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        y_tensor_list = []

        self.max_len = self.get_max_len()

        if self.type == 1:
            for example in data:
                label, _, sentence, _ = example
                encoded_dict = self.tokenizer.encode_plus(
                                            sentence, 
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
                y_tensor = torch.tensor(self.y_mapping[label])
                y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        elif self.type == 2:
            for example in data:
                label, _, sentence_1, sentence_2 = example
                encoded_dict = self.tokenizer.encode_plus(
                                            sentence_1, 
                                            sentence_2,
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
                y_tensor = torch.tensor(self.y_mapping[label])
                y_tensor_list.append(torch.unsqueeze(y_tensor, dim=0))
        elif self.type == "qa":
            for example in data:
                start_span, end_span, question, context = example
                start_span = int(start_span) + 1
                end_span = int(end_span) + 1
                if len(self.tokenizer.tokenize(question)) + len(self.tokenizer.tokenize(context)) + 3 > 512:
                    print("")
                    print("ATTENTION: example too long!")
                    print("question: {}".format(question))
                    print("context: {}".format(context))
                    continue
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

        self.make_datasets()


###############################
######## d e I S E A R ########

class deISEAR_dataloader(Dataloader):
    def load(self):
        self.type = 1
        self.path_dev = self.path + "GLIBERT_deISEAR_dev.tsv"
        self.path_test = self.path + "GLIBERT_deISEAR_test.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)
    
####################################
########### M L Q A ############

class MLQA_dataloader(Dataloader):
    def load(self):
        self.type = "qa"
        self.path_dev = str(Path(self.path)) + "/dev/GLIBERT_dev-context-de-question-de.tsv"
        self.path_test = str(Path(self.path)) + "/test/GLIBERT_test-context-de-question-de.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

####################################
########### P A W S - X ############

class PAWS_X_dataloader(Dataloader):
    def load(self):
        self.type = 2
        self.path_dev = str(Path(self.path)) + "/de/GLIBERT_paws_x_dev.tsv"
        self.path_test = str(Path(self.path)) + "/de/GLIBERT_paws_x_test.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

###########################
######## S C A R E ########

class SCARE_dataloader(Dataloader):
    def load(self):
        self.type = 1
        self.path_dev = str(Path(self.path)) + "/scare_v1.0.0/annotations/GLIBERT_annotations_dev.tsv"
        self.path_test = str(Path(self.path)) + "/scare_v1.0.0/annotations/GLIBERT_annotations_test.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

########################
######## X N L I #######

class XNLI_dataloader(Dataloader):
    def load(self):
        self.type = 2
        self.path_dev = str(Path(self.path)) + "/XNLI-1.0/GLIBERT_xnli.dev.de.tsv"
        self.path_test = str(Path(self.path)) + "/XNLI-1.0/GLIBERT_xnli.test.de.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

##################################
########### X Q u A D ############

class XQuAD_dataloader(Dataloader):
    def load(self):
        self.type = "qa"
        self.path_dev = str(Path(self.path)) + "/xquad/GLIBERT_xquad_dev.tsv"
        self.path_test = str(Path(self.path)) + "/xquad/GLIBERT_xquad_test.tsv"
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)


#################################

def dataloader(config, location, data_set):
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
    num_classes = len(dataloader.y_mapping) if dataloader.y_mapping else dataloader.max_len
    mapping = dataloader.y_mapping
    max_len = dataloader.max_len
    data_type = dataloader.type

    return dev_dataloader, test_dataloader, num_classes, max_len, mapping, data_type

