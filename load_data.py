import csv
import torch

from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer


class gliBertDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, attention_masks, token_type_ids, srl, ids):
        assert all(len(obj) == len(x_tensor) for obj in [y_tensor, attention_masks, token_type_ids, srl])
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.srl = srl
        self.ids = ids

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_tensor[idx], \
                self.y_tensor[idx], \
                self.attention_masks[idx], \
                self.token_type_ids[idx], \
                self.srl[idx], \
                self.ids[idx]


class Dataloader:
    def __init__(self, path_data, path_tokenizer, batch_size, merge_subtokens, max_length):
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.merge_subtokens = merge_subtokens
        self.batch_size = batch_size
        self.path = path_data
        self.max_length = max_length
        self.path_train = None
        self.path_dev = None
        self.path_test = None
        self.data_train = None
        self.data_dev = None
        self.data_test = None
        self.max_len = None
        self.type = None
        self.attention_mask_train = None
        self.attention_mask_dev = None
        self.attention_mask_test = None
        self.token_type_ids_train = None
        self.token_type_ids_dev = None
        self.token_type_ids_test = None
        self.srl_train = None
        self.srl_dev = None
        self.srl_test = None
        self.x_tensor_train = None
        self.x_tensor_dev = None
        self.x_tensor_test = None
        self.y_tensor_train = None
        self.y_tensor_dev = None
        self.y_tensor_test = None
        self.y_mapping = {}
        self.dataset_train = None
        self.dataset_dev = None
        self.dataset_test = None

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

    def check_max_length(self, *sent_lengths):
        to_add = 1 if len(sent_lengths) == 1 else 2
        max_length = 0
        for sent_length in sent_lengths:
            max_length += sent_length
        max_length += to_add
        return max_length if max_length < self.max_length else self.max_length

    def load_data(self, path):
        data = []
        y_mapping = self.y_mapping
        with open(path, "r") as f:
            f_reader = csv.reader(f, delimiter="\t")
            counter = 0
            for row in f_reader:
                if self.type == 1:
                    instance_id, label, blank, sentence, srl_sentence = row[0], row[1], row[2], row[3], row[4]
                    srl_sentence = eval(srl_sentence)
                    data.append((instance_id, label, blank, sentence, "", srl_sentence, ""))
                elif self.type == 2:
                    instance_id, \
                    label, \
                    blank, \
                    sentence_1, \
                    sentence_2, \
                    srl_sentence_1, \
                    srl_sentence_2 = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                    srl_sentence_1 = eval(srl_sentence_1)
                    srl_sentence_2 = eval(srl_sentence_2)
                    data.append((instance_id, \
                                    label, \
                                    blank, \
                                    sentence_1, \
                                    sentence_2, \
                                    srl_sentence_1, \
                                    srl_sentence_2
                                    ))
                elif self.type == "qa":
                    instance_id, \
                    start_index, \
                    text, \
                    context, \
                    question, \
                    srl_context, \
                    srl_question = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                    start_index = int(start_index)
                    srl_context = eval(srl_context)
                    srl_question = eval(srl_question)
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
                    data.append((instance_id, start_span, end_span, question, context, srl_question, srl_context))

                if self.type != "qa":
                    if label not in y_mapping:
                        y_mapping[label] = counter
                        counter += 1
    
        self.y_mapping = y_mapping
        return data

    def make_datasets(self):
        self.dataset_train = gliBertDataset(
                                self.x_tensor_train,
                                self.y_tensor_train,
                                self.attention_mask_train,
                                self.token_type_ids_train,
                                self.srl_train,
                                self.ids_train
                                )
        self.dataset_dev = gliBertDataset(
                                self.x_tensor_dev,
                                self.y_tensor_dev,
                                self.attention_mask_dev,
                                self.token_type_ids_dev,
                                self.srl_dev,
                                self.ids_dev
                                )
        self.dataset_test = gliBertDataset(
                                self.x_tensor_test,
                                self.y_tensor_test,
                                self.attention_mask_test,
                                self.token_type_ids_test,
                                self.srl_test,
                                self.ids_test
                                )

    def get_max_len(self):
        if self.type == 2 or self.type == "qa":
            longest_sentence_1_train = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_train]) 
            longest_sentence_2_train = max([len(self.tokenizer.tokenize(sent[4])) for sent in self.data_train]) 
            longest_sentence_1_dev = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_dev]) 
            longest_sentence_2_dev = max([len(self.tokenizer.tokenize(sent[4])) for sent in self.data_dev]) 
            longest_sentence_1_test = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_test]) 
            longest_sentence_2_test = max([len(self.tokenizer.tokenize(sent[4])) for sent in self.data_test]) 
            longest_sentence_1 = max(longest_sentence_1_train, longest_sentence_1_dev, longest_sentence_1_test)
            longest_sentence_2 = max(longest_sentence_2_train, longest_sentence_2_dev, longest_sentence_2_test)
            return self.check_max_length(longest_sentence_1, longest_sentence_2)
        else:
            longest_sent_train = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_train]) 
            longest_sent_dev = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_dev]) 
            longest_sent_test = max([len(self.tokenizer.tokenize(sent[3])) for sent in self.data_test]) 
            return self.check_max_length(max(longest_sent_train, longest_sent_dev, longest_sent_test))

    def load_torch_data(self, data):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        y_tensor_list = []
        srl = []
        ids = []

        self.max_len = self.get_max_len()

        if self.type == 1:
            for example in data:
                instance_id, label, _, sentence, _ , srl_sentence, _ = example
                if len(self.tokenizer.tokenize(sentence)) + \
                        2 > self.max_length:
                    print("")
                    print("ATTENTION: example too long!")
                    print("sentence: {}".format(sentence))
                    continue
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
                srl.append([srl_sentence])
                ids.append(instance_id)
        elif self.type == 2:
            for example in data:
                instance_id, label, _, sentence_1, sentence_2, srl_sentence_1, srl_sentence_2 = example
                if len(self.tokenizer.tokenize(sentence_1)) + \
                        len(self.tokenizer.tokenize(sentence_2)) + \
                        3 > self.max_length:
                    print("")
                    print("ATTENTION: example too long!")
                    print("sentence 1: {}".format(sentence_1))
                    print("sentence 2: {}".format(sentence_2))
                    continue
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
                srl.append([srl_sentence_1, srl_sentence_2])
                ids.append(instance_id)
        elif self.type == "qa":
            for example in data:
                instance_id, start_span, end_span, question, context, srl_question, srl_context = example
                start_span = int(start_span) + 1
                end_span = int(end_span) + 1
                if len(self.tokenizer.tokenize(question)) + \
                        len(self.tokenizer.tokenize(context)) + \
                        3 > self.max_length:
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
                srl.append([srl_question, srl_context])
                ids.append(instance_id)
            
        return torch.cat(input_ids, dim=0), \
                torch.cat(attention_mask, dim=0), \
                torch.cat(token_type_ids, dim=0), \
                torch.cat(tuple(y_tensor_list), dim=0), \
                srl, \
                ids

    def load_torch(self):
        self.x_tensor_train, \
        self.attention_mask_train, \
        self.token_type_ids_train, \
        self.y_tensor_train, \
        self.srl_train, \
        self.ids_train = self.load_torch_data(self.data_train)

        self.x_tensor_dev, \
        self.attention_mask_dev, \
        self.token_type_ids_dev, \
        self.y_tensor_dev, \
        self.srl_dev, \
        self.ids_dev = self.load_torch_data(self.data_dev)

        self.x_tensor_test, \
        self.attention_mask_test, \
        self.token_type_ids_test, \
        self.y_tensor_test, \
        self.srl_test, \
        self.ids_test = self.load_torch_data(self.data_test)

        print("")
        print("======== Longest example in data: ========")
        print("length (tokenized): {}".format(self.max_len))

        self.make_datasets()


###############################
######## d e I S E A R ########

class deISEAR_dataloader(Dataloader):
    def load(self):
        self.type = 1
        self.path_train = str(Path(self.path)) + "/gliBert_deISEAR_train.tsv"
        self.path_dev = str(Path(self.path)) + "/gliBert_deISEAR_dev.tsv"
        self.path_test = str(Path(self.path)) + "/gliBert_deISEAR_test.tsv"
        self.data_train = self.load_data(self.path_train)
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)
    
################################
########### M L Q A ############

class MLQA_dataloader(Dataloader):
    def load(self):
        self.type = "qa"
        self.path_train = str(Path(self.path)) + "/gliBert_mlqa_train.tsv"
        self.path_dev = str(Path(self.path)) + "/gliBert_mlqa_dev.tsv"
        self.path_test = str(Path(self.path)) + "/gliBert_mlqa_test.tsv"
        self.data_train = self.load_data(self.path_train)
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

####################################
########### P A W S - X ############

class PAWS_X_dataloader(Dataloader):
    def load(self):
        self.type = 2
        self.path_train = str(Path(self.path)) + "/gliBert_paws_x_train.tsv"
        self.path_dev = str(Path(self.path)) + "/gliBert_paws_x_dev.tsv"
        self.path_test = str(Path(self.path)) + "/gliBert_paws_x_test.tsv"
        self.data_train = self.load_data(self.path_train)
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

###########################
######## S C A R E ########

class SCARE_dataloader(Dataloader):
    def load(self):
        self.type = 1
        self.path_train = str(Path(self.path)) + "/gliBert_scare_annotations_train.tsv"
        self.path_dev = str(Path(self.path)) + "/gliBert_scare_annotations_dev.tsv"
        self.path_test = str(Path(self.path)) + "/gliBert_scare_annotations_test.tsv"
        self.data_train = self.load_data(self.path_train)
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

########################
######## X N L I #######

class XNLI_dataloader(Dataloader):
    def load(self):
        self.type = 2
        self.path_train = str(Path(self.path)) + "/gliBert_xnli_train.tsv"
        self.path_dev = str(Path(self.path)) + "/gliBert_xnli_dev.tsv"
        self.path_test = str(Path(self.path)) + "/gliBert_xnli_test.tsv"
        self.data_train = self.load_data(self.path_train)
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)

##################################
########### X Q u A D ############

class XQuAD_dataloader(Dataloader):
    def load(self):
        self.type = "qa"
        self.path_train = str(Path(self.path)) + "/gliBert_xquad_train.tsv"
        self.path_dev = str(Path(self.path)) + "/gliBert_xquad_dev.tsv"
        self.path_test = str(Path(self.path)) + "/gliBert_xquad_test.tsv"
        self.data_train = self.load_data(self.path_train)
        self.data_dev = self.load_data(self.path_dev)
        self.data_test = self.load_data(self.path_test)


#################################

def dataloader(config, location, data_set):
    if data_set == "deISEAR":
        dataloader = deISEAR_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"],
                            config["max_length"]
                            )
    elif data_set == "MLQA":
        dataloader = MLQA_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"],
                            config["max_length"]
                            )
    elif data_set == "PAWS-X":
        dataloader = PAWS_X_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"],
                            config["max_length"]
                            )
    elif data_set == "SCARE":
        dataloader = SCARE_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"],
                            config["max_length"]
                            )
    elif data_set == "XNLI":
        dataloader = XNLI_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"],
                            config["max_length"]
                            )
    elif data_set == "XQuAD":
        dataloader = XQuAD_dataloader(
                            config[location][data_set],
                            config[location]["BERT"],
                            config["batch_size"],
                            config["merge_subtokens"],
                            config["max_length"]
                            )

    dataloader.load()
    dataloader.load_torch()
    num_classes = len(dataloader.y_mapping) if dataloader.y_mapping else dataloader.max_len

    return dataloader.dataset_train, \
            dataloader.dataset_dev, \
            dataloader.dataset_test, \
            num_classes, \
            dataloader.max_len, \
            dataloader.y_mapping, \
            dataloader.type

