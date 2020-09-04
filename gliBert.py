import argparse
import datetime
import json
import time
import torch

import numpy as np

from load_data import Dataloader, dataloader
from random import shuffle
from torch import nn
from transformers import (
        AdamW,
        BertModel,
        BertTokenizer,
        get_linear_schedule_with_warmup
        )

merge_subs = Dataloader.merge_subs


blue = "\033[94m"

green = "\033[92m"

red = "\033[93m"

end = "\033[0m"


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-c", 
            "--config", 
            type=str, 
            help="Path to hyperparamter/config file (json)"
            )
    parser.add_argument(
            "-d", 
            "--data_set", 
            type=str, 
            help="Indicate on which data set model should be trained",
            choices=["deISEAR", "XNLI", "MLQA", "SCARE", "PAWS-X", "XQuAD"]
            )
    parser.add_argument(
            "-l", 
            "--location", 
            type=str, 
            help="Indicate where model will be trained",
            choices=["local", "rattle"]
            )
    parser.add_argument(
            "-s", 
            "--stats_file", 
            type=str, 
            help="Specify file for writing stats to",
            )
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, "r") as f:
        f = f.read()
        data = json.loads(f)
    return data


def sigmoid(x):
       return 1/(1+np.exp(-x))


def swish(x):
        return x * sigmoid(x)


def Swish(batch):
    swish_tensors = []
    for tensor in batch:
        swish_tensors.append(torch.tensor(list(map(swish, tensor))))
    return torch.stack(tuple(batch))


def pad_SRLs(batch, dummy, length):
    """
    Pad a batch of token SRLs to tokenized batch
    """
    new_batch = []
    for example in batch:
        lst = [example] + [dummy]*(length-len(example))
        tens = torch.cat(tuple(lst), dim=0) 
        new_batch.append(tens)

    new_batch = torch.stack(new_batch)

    return new_batch


class SRL_Encoder(nn.Module):
    def __init__(self, config):
        super(SRL_Encoder, self).__init__()
        self.config = config
        self.dictionary = {
            "B-A0":	0,
            "B-A1":	1,
            "B-A2":	2,
            "B-A3":	3,
            "B-A4":	4,
            "B-A5":	5,
            "B-A6":	6,
            "B-A7":	7,
            "B-A8":	8,
            "B-A9":	9,
            "B-C-A0":	10,
            "B-C-A1":	11,
            "B-C-A2":	12,
            "B-C-A3":	13,
            "B-C-A4":	14,
            "B-C-A5":	15,
            "B-C-A6":	16,
            "B-C-A7":	17,
            "B-C-A8":	18,
            "B-C-C-A0":	19,
            "B-C-C-A1":	20,
            "B-C-C-A2":	21,
            "B-V":	22,
            "I-A0":	23,
            "I-A1":	24,
            "I-A2":	25,
            "I-A3":	26,
            "I-A4":	27,
            "I-A5":	28,
            "I-A6":	29,
            "I-A7":	30,
            "I-A8":	31,
            "I-A9":	32,
            "I-C-A0":	33,
            "I-C-A1":	34,
            "I-C-A2":	35,
            "I-C-A3":	36,
            "I-C-A4":	37,
            "I-C-A5":	38,
            "I-C-A6":	39,
            "I-C-A7":	40,
            "I-C-A8":	41,
            "I-C-C-A0":	42,
            "I-C-C-A1":	43,
            "O":	44,
            "0":        45
        }
        self.embeddings = nn.Embedding(len(self.dictionary), self.config["embedding_dim"])
        self.encoder = nn.GRU(
                            input_size=3*self.config["embedding_dim"],
                            hidden_size=self.config["gru_hidden_size"],
                            num_layers=self.config["num_layers"],
                            bias=self.config["bias"],
                            batch_first=True,
                            dropout=self.config["gru_dropout"],
                            bidirectional=self.config["bidirectional"]
                            )

    def forward(self, tokens):
        """
        forward pass, tokens come in shape: batch:sentence:predicate:ids
        Args:
            param1: list[list[list[list]]]

        Returns:
            list
        """
        batch_new = []
        for batch in tokens:
            sentences = []
            for sentence in batch:
                num_preds = len(sentence)
                pred_1 = self.embeddings(sentence[0])
                pred_2 = self.embeddings(sentence[1]) if num_preds > 1 else pred_1
                pred_3 = self.embeddings(sentence[2]) if num_preds > 2 else pred_1
                pred_merge = []
                for i in range(len(pred_1)):
                    tokens = [pred_1[i], pred_2[i], pred_3[i]]
                    pred_merge.append(torch.cat(tuple(tokens), dim=0))
                preds = torch.unsqueeze(torch.stack(pred_merge), dim=0)

                output, _ = self.encoder(preds)
                sentences.append(torch.squeeze(output, dim=0))
            sequence = torch.cat(tuple(sentences), dim=0)
            batch_new.append(sequence)

        #torch.stack(batch_new)

        return batch_new


class BertBase(nn.Module):
    @staticmethod
    def ensure_end_span_behind_start_span(batch_start_tensor, batch_end_tensor, device):
        """Set all probabilities up to start span to -inf for end spans.
        Args:
            param1: tensor[tensor]
            param2: tensor[tensor]
        Returns:
            tensor[tensor]
        """
        new_batch_tensor = []
        
        for i, end_tensor in enumerate(batch_end_tensor):
            start_index = batch_start_tensor[i].max(0).indices.item()
            new_end_tensor = torch.cat(
                                tuple([
                                    torch.tensor([float("-inf")]*start_index).to(device),
                                    end_tensor[start_index:]])
                                )
            new_batch_tensor.append(new_end_tensor)

        return torch.stack(new_batch_tensor)

    def reconstruct_word_level(self, batch, ids, device="cpu"):
        """method for joining subtokenized words back to word level
        The idea is to average subtoken embeddings.
        To preserve original length, append last subtoken-embedding (which
        is per definition [PAD] since padding + 1 of max-length) until original
        length is reached.

        Args:
            param1: torch.tensor embeddings of subtokens
            param2: torch.tensor indices of subtokens
        Returns:
            torch.tensor of same input dimensions with averaged embeddings
                        for subtokens
        """
        pad_token = torch.tensor([0.]*768).to(device)
        word_level_batch = []
        batch_idx = []
        
        for j, sentence in enumerate(batch):
            A_sent_idx = []
            B_sent_idx = []
            first_second = A_sent_idx
            word_level_sentence = []
            for i, token in enumerate(sentence):
                decode_token = self.tokenizer.decode([ids[j][i]])
                if decode_token.startswith("##"):
                    continue
                elif decode_token in ["[CLS]", "[UNK]"]:
                    word_level_sentence.append(token)
                elif decode_token == "[MASK]":
                    word_level_sentence.append(token)
                    first_second.append(0)
                elif decode_token == "[SEP]":
                    first_second = B_sent_idx
                    if i + 1 == len(sentence):
                        break
                elif decode_token == "[PAD]":
                    break
                elif not self.tokenizer.decode([ids[j][i+1]]).startswith("##"):
                    word_level_sentence.append(token)
                    first_second.append(0)
                else:
                    current_word = [token]
                    split_counter = 0
                    for k, subtoken in enumerate(sentence[i+1:]):
                        decode_subtoken = self.tokenizer.decode([ids[j][i+k+1]])
                        if decode_subtoken.startswith("##"):
                            current_word.append(subtoken)
                            split_counter += 1
                        else:
                            break
                    current_word = torch.stack(tuple(current_word))
                    mean_embs_word = torch.mean(current_word, 0)
                    word_level_sentence.append(mean_embs_word)
                    first_second.append(split_counter)
            #pad_token = sentence[-1]
            while len(word_level_sentence) < len(sentence):
                word_level_sentence.append(pad_token)
            word_level_batch.append(torch.stack(tuple(word_level_sentence)))
            batch_idx.append((A_sent_idx, B_sent_idx))

        return_batch = torch.stack(tuple(word_level_batch))
        
        return return_batch, batch_idx


class BertClassifierCLS(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config[location]["BERT"])
        self.tokenizer = BertTokenizer.from_pretrained(self.config[location]["BERT"])
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            srls=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        _, pooler_output = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        linear_output = self.linear(pooler_output)
        proba = self.softmax(linear_output)
        return proba


class BertClassifierLastHiddenStateAll(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.tokenizer = BertTokenizer.from_pretrained(self.config[location]["BERT"])
        self.linear = nn.Linear(768*max_len, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            srls=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        last_hidden_state, _ = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        if self.config["merge_subtokens"] == True:
            full_word_hidden_state, _ = self.reconstruct_word_level(last_hidden_state, tokens)
        reshaped_last_hidden = torch.reshape(
                full_word_hidden_state if self.config["merge_subtokens"] == True else last_hidden_state, 
                (
                    last_hidden_state.shape[0], 
                    last_hidden_state.shape[1]*last_hidden_state.shape[2])
                )
        linear_output = self.linear(reshaped_last_hidden)
        proba = self.softmax(linear_output)
        return proba


class gliBertClassifierLastHiddenStateAll(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        self.linear = nn.Linear((768+2*config["gru_hidden_size"])*max_len, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            srls=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        # for glueing srls AB together and padding. Has to be double since
        # bi-directional. size batch:1:2*hidden_size
        dummy_srl = torch.tensor([0.0]*2*self.config["gru_hidden_size"]).to(device)
        dummy_srl = torch.unsqueeze(dummy_srl, dim=0)

        last_hidden_state, _ = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        if data_type != 1:
            a_srls, b_srls = get_AB_SRLs(srls) 
            a_emb = self.srl_model(a_srls)
            b_emb = self.srl_model(b_srls)
            ab = lambda i: [dummy_srl, a_emb[i], dummy_srl, b_emb[i]]
            srl_emb = [torch.cat(tuple(ab(i)), dim=0) for i in range(len(a_srls))]
        else:
            srl_emb = self.srl_model(get_A_SRLs(srls))

        if self.config["merge_subtokens"] == True:
            full_word_hidden_state, split_idxs = self.reconstruct_word_level(
                                                            last_hidden_state, \
                                                            tokens, \
                                                            device
                                                            )
            srl_batch = pad_SRLs(srl_emb, dummy_srl, self.max_len)
            combo_merge_batch = torch.cat(tuple([full_word_hidden_state, srl_batch]), dim=-1)

        reshaped_last_hidden = torch.reshape(
                combo_merge_batch if self.config["merge_subtokens"] == True else last_hidden_state, 
                (
                    combo_merge_batch.shape[0], 
                    combo_merge_batch.shape[1]*combo_merge_batch.shape[2])
                )
        import ipdb; ipdb.set_trace()
        linear_output = self.linear(reshaped_last_hidden)
        proba = self.softmax(linear_output)
        return proba


# class gliBertClassifierLastHiddenStateAllGRU(BertBase):
#     def __init__(self, config, num_classes, max_len):
#         super(BertBase, self).__init__()
#         self.config = config
#         self.bert = BertModel.from_pretrained(config[location]["BERT"])
#         self.srl_model = SRL_Encoder(config)
#         self.max_len = max_len
#         self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
#         self.linear = nn.Linear((768+2*config["gru_hidden_size"])*max_len, num_classes)
#         self.softmax = nn.LogSoftmax(dim=-1)
#     
#     def forward(
#             self,
#             tokens,
#             attention_mask=None,
#             token_type_ids=None,
#             srls=None,
#             data_type=None,
#             device=torch.device("cpu")
#             ):
#         # for glueing srls AB together and padding. Has to be double since
#         # bi-directional. size batch:1:2*hidden_size
#         dummy_srl = torch.tensor([0.0]*2*self.config["gru_hidden_size"]).to(device)
#         dummy_srl = torch.unsqueeze(dummy_srl, dim=0)
# 
#         last_hidden_state, _ = self.bert(
#                                 tokens,
#                                 attention_mask,
#                                 token_type_ids
#                                 )
#         if data_type != 1:
#             a_srls, b_srls = get_AB_SRLs(srls) 
#             a_emb = self.srl_model(a_srls)
#             b_emb = self.srl_model(b_srls)
#             ab = lambda i: [dummy_srl, a_emb[i], dummy_srl, b_emb[i]]
#             srl_emb = [torch.cat(tuple(ab(i)), dim=0) for i in range(len(a_srls))]
#         else:
#             srl_emb = self.srl_model(get_A_SRLs(srls))
# 
#         if self.config["merge_subtokens"] == True:
#             full_word_hidden_state = self.reconstruct_word_level(last_hidden_state, tokens) 
#             srl_batch = pad_SRLs(srl_emb, dummy_srl, self.max_len)
#             combo_merge_batch = torch.cat(tuple([full_word_hidden_state, srl_batch]), dim=-1)
# 
#         reshaped_last_hidden = torch.reshape(
#                 combo_merge_batch if self.config["merge_subtokens"] == True else last_hidden_state, 
#                 (
#                     combo_merge_batch.shape[0], 
#                     combo_merge_batch.shape[1]*combo_merge_batch.shape[2])
#                 )
#         #import ipdb; ipdb.set_trace()
#         linear_output = self.linear(reshaped_last_hidden)
#         proba = self.softmax(linear_output)
#         return proba


class BertClassifierLastHiddenStateNoCLS(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.tokenizer = BertTokenizer.from_pretrained(self.config[location]["BERT"])
        self.linear = nn.Linear(768*(max_len-1), num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            srls=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        last_hidden_state, _ = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        last_hidden_state = last_hidden_state[:, 1:, :]
        if self.config["merge_subtokens"] == True:
            full_word_hidden_state = self.reconstruct_word_level(last_hidden_state, tokens) 
        reshaped_last_hidden = torch.reshape(
                full_word_hidden_state if self.config["merge_subtokens"] == True else last_hidden_state, 
                (
                    last_hidden_state.shape[0], 
                    last_hidden_state.shape[1]*last_hidden_state.shape[2])
                )
        linear_output = self.linear(reshaped_last_hidden)
        proba = self.softmax(linear_output)
        return proba


class gliBertClassifierLastHiddenStateNoCLS(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        self.linear = nn.Linear((768+2*config["gru_hidden_size"])*(max_len-1), num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            srls=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        # for glueing srls AB together and padding. Has to be double since
        # bi-directional. size batch:1:2*hidden_size
        dummy_srl = torch.tensor([0.0]*2*self.config["gru_hidden_size"]).to(device)
        dummy_srl = torch.unsqueeze(dummy_srl, dim=0)

        last_hidden_state, _ = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        last_hidden_state = last_hidden_state[:, 1:, :]
        if data_type != 1:
            a_srls, b_srls = get_AB_SRLs(srls) 
            a_emb = self.srl_model(a_srls)
            b_emb = self.srl_model(b_srls)
            ab = lambda i: [a_emb[i], dummy_srl, b_emb[i]]
            srl_emb = [torch.cat(tuple(ab(i)), dim=0) for i in range(len(a_srls))]
        else:
            srl_emb = self.srl_model(get_A_SRLs(srls))

        if self.config["merge_subtokens"] == True:
            full_word_hidden_state = self.reconstruct_word_level(last_hidden_state, tokens) 
            srl_batch = pad_SRLs(srl_emb, dummy_srl, self.max_len-1)
            combo_merge_batch = torch.cat(tuple([full_word_hidden_state, srl_batch]), dim=-1)

        reshaped_last_hidden = torch.reshape(
                combo_merge_batch if self.config["merge_subtokens"] == True else last_hidden_state, 
                (
                    combo_merge_batch.shape[0], 
                    combo_merge_batch.shape[1]*combo_merge_batch.shape[2])
                )
        #import ipdb; ipdb.set_trace()
        linear_output = self.linear(reshaped_last_hidden)
        proba = self.softmax(linear_output)
        return proba


class BertSpanPrediction(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.tokenizer = BertTokenizer.from_pretrained(self.config[location]["BERT"])
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(dim=-2)
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            srls=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        last_hidden_state, _ = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        if self.config["merge_subtokens"] == True:
            full_word_hidden_state = self.reconstruct_word_level(last_hidden_state, tokens) 
        linear_output = self.linear(last_hidden_state)
        start_logits, end_logits = linear_output.split(1, dim=-1)
        start_span = self.softmax(start_logits)
        end_span = self.softmax(end_logits)
        return start_span, end_span


def combine_srl_embs_bert_embs():
    """
    """
    comb_tensor = torch.cat((bert_tensor, srl_tensor), 0)
    pass


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def compute_acc(preds, labels):
    """computes the accordance of two lists
    Args:
        param1: list
        param2: list
    Returns:
        float
    """
    correct = 0
    assert len(preds) == len(labels)
    for pred, lab in zip(preds, labels):
        if pred == lab: correct += 1
    return correct / len(preds)


def write_stats(stats_file, training_stats, training_results):
    """checks if outfile specified, if yes, writes stats to it
    Args:
        prarm1: str
    Returns:
        None
    """
    if stats_file is not None:
        stamp = datetime.datetime.now().strftime("--%d-%m-%Y_%H-%M-%S")
        if "." in stats_file:
            name, ext = stats_file.split(".")
            stats_file = name + stamp + ext
        else:
            stats_file = stats_file + stamp + ".json"
        with open(stats_file, "w") as outfile:
            outfile.write(json.dumps(training_stats))

        results_file = stats_file.rstrip(".json") + ".results.json"
        with open(results_file, "w") as outfile:
            outfile.write(json.dumps(training_results))


def print_preds(model, \
                data_type, \
                example, \
                srls, \
                prediction, \
                true_label, \
                mapping, \
                step, \
                len_data, \
                elapsed, merge):
    first_srls = [sentence[0][0].tolist() for sentence in srls]
    reverse_dict = {value: key for key, value in model.srl_model.dictionary.items()}
    if not data_type == "qa":
        print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len_data, elapsed))
        print("  Last prediction: ")
        #print("    Text:   {}".format(model.tokenizer.decode(example, skip_special_tokens=True)))
        print("    Text:   {}".format("\t".join(merge_subs([x for x in model.tokenizer.tokenize(model.tokenizer.decode(example, skip_special_tokens=True))]))))
        print("    SRLs:  {}".format("\t".join([reverse_dict[srl] for ls in first_srls for srl in ls])))
        print("    Prediction:  {}".format(mapping[prediction.max(0).indices.item()]))
        print("    True Label:  {}".format(mapping[true_label.item()]))
        print("")
    else:
        start_span, end_span = prediction
        print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len_data, elapsed))
        print("  Last prediction: ")
        print("    Text:   {}".format(model.tokenizer.decode(example, skip_special_tokens=True)))
        if not merge:
            sentence = model.tokenizer.tokenize(
                                model.tokenizer.decode(
                                    example
                                    )
                                )
        else:
            sentence = merge_subs(model.tokenizer.tokenize(
                                model.tokenizer.decode(
                                    example
                                    )
                                ))
        prediction = sentence[start_span.max(0).indices.item():end_span.max(0).indices.item()+1] 
        true_span = sentence[true_label.select(0, 0).item():true_label.select(0, 1).item()+1]
        print("    Prediction:  {}".format(" ".join(prediction)))
        print("    True Span:  {}".format(" ".join(true_span)))
        print("")


def batch_idcs(len_dataset, batch_size):
    batch_idcs = []
    current_idx = 0
    while len_dataset > current_idx:
        batch_idcs.append((current_idx, current_idx + batch_size))
        current_idx += batch_size

    shuffle(batch_idcs)
    return batch_idcs


def convert_SRLs_to_tensor(mapping, lst, device="cpu"):
    """
    turns a nested list of SRLs into a list of tensors of indices of SRLs
    Args:
        param1: dict    mapping of SRLs to indices
        prarm2: list[list[list[list[list]]]]    batch:AB:sentences:predicates:SRLs
        param3: torch.device
    Return:
        list[list[list[list[torch.tensor]]]]
    """
    new_lst = []
    for batch in lst:
        new_batch = []
        for AB in batch:
            new_AB = []
            for sentence in AB:
                new_sentence = []
                for predicate in sentence:
                    new_predicate = [mapping[srl] for srl in predicate]
                    new_predicate = torch.tensor(new_predicate).to(device)
                    new_sentence.append(new_predicate)
                new_AB.append(new_sentence)
            new_batch.append(new_AB)
        new_lst.append(new_batch)
    return new_lst


def get_AB_SRLs(lst):
    a_lst, b_lst = ([] for i in range(2))
    for batch in lst:
        a_lst.append(batch[0])
        b_lst.append(batch[1])
    return a_lst, b_lst


def get_A_SRLs(lst):
    a_lst = []
    for batch in lst:
        a_lst.append(batch[0])
    return a_lst


def fine_tune_BERT(config):
    """define fine-tuning procedure, write results to file.
    Args:
        param1: nn.Model (BERT-model)
        param2: torch.tensor
        param3: dict
    Returns:
        None
    """
    epochs = config["epochs"]
    gpu = config["gpu"]
    batch_size = config["batch_size"]
    print_stats = config["print_stats"]
    bert_head = eval(config["bert_head"])
    criterion = nn.NLLLoss()
    merge_subtokens = config["merge_subtokens"]

    train_data, \
    dev_data, \
    test_data, \
    num_classes, \
    max_len, \
    mapping, \
    data_type = dataloader(config, location, data_set)
    mapping = {value: key for (key, value) in mapping.items()} if mapping else None

    model = bert_head(config, num_classes, max_len)
    train_idcs = batch_idcs(len(train_data), batch_size)
    dev_idcs = batch_idcs(len(dev_data), batch_size)
    test_idcs = batch_idcs(len(test_data), batch_size)

    print("")
    print("======== Checking which device to use... ========")
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
        model.cuda(device)
        print("")
        print(">>  device set to: CUDA -> using GPU #{}".format(gpu))
    else:
        device = torch.device("cpu")
        print("")
        print(">>  device set to: CPU")

    optimizer = AdamW(
            model.parameters(),
            lr = 2e-5,
            eps = 1e-8
        )

    total_steps = len(dev_data) * epochs
    scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = 0,
            num_training_steps = total_steps
        )
    training_stats = []
    training_stats.append(config)
    results = []
    total_t0 = time.time()
    PATIENCE = 0

    for epoch_i in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print(blue + "Training..." + end)
        t0 = time.time()
        total_train_loss = 0

        #################
        ### Train Run ###

        model.train()
        for step, idcs in enumerate(train_idcs):
            batch = train_data[idcs[0]:idcs[1]]
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            b_srls = batch[4]
            b_srls_idx = convert_SRLs_to_tensor(model.srl_model.dictionary, b_srls, device)
            model.zero_grad()
            if not data_type == "qa":
                outputs = model(
                            b_input_ids, 
                            attention_mask=b_attention_mask,
                            token_type_ids=b_token_type_ids,
                            srls = b_srls_idx,
                            device=device,
                            data_type=data_type
                            )
                if step % print_stats == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print_preds(
                            model,
                            data_type,
                            b_input_ids[-1],
                            b_srls_idx[-1],
                            outputs[-1],
                            b_labels[-1],
                            mapping,
                            step,
                            len(train_idcs),
                            elapsed,
                            merge_subtokens
                            )
                loss = criterion(outputs, b_labels)
            else:
                start_span, end_span = model(
                                        b_input_ids,
                                        attention_mask=b_attention_mask,
                                        token_type_ids=b_token_type_ids,
                                        srls = b_srls_idx,
                                        device=device,
                                        data_type=data_type
                                        )
                if step % print_stats == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print_preds(
                            model,
                            data_type,
                            b_input_ids[-1],
                            b_srls_idx[-1],
                            (start_span[-1], end_span[-1]),
                            b_labels[-1],
                            mapping, step,
                            len(train_idcs),
                            elapsed,
                            merge_subtokens
                            )
                start_loss = criterion(start_span, torch.unsqueeze(b_labels.select(1, 0), -1))
                end_loss = criterion(end_span, torch.unsqueeze(b_labels.select(1, 1), -1))
                loss = (start_loss + end_loss) / 2
            total_train_loss += loss.item()
            loss.backward()
            # This is to help prevent the "exploding gradients" problem. (Maybe not necessary?)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(dev_data)
        train_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(train_time))

        ###############
        ### Dev Run ###

        print("")
        print(green + "Running Development Set evaluation..." + end)
        t0 = time.time()
        model.eval()

        dev_results = []
        if epoch_i == 0:
            results.append(
                    {"RESULTS": "id, prediction, gold"}
            )

        total_dev_accuracy = 0
        total_dev_loss = 0

        for idcs in dev_idcs:
            batch = dev_data[idcs[0]:idcs[1]]
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            b_srls = batch[4]
            b_ids = batch[5]
            b_srls_idx = convert_SRLs_to_tensor(model.srl_model.dictionary, b_srls, device)


            with torch.no_grad():
                if not data_type == "qa":
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_attention_mask,
                                token_type_ids=b_token_type_ids,
                                srls = b_srls_idx,
                                device=device,
                                data_type=data_type
                                )
                    value_index = [tensor.max(0) for tensor in outputs]
                    acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                    loss = criterion(outputs, b_labels)

                    preds = [mapping[maxs.indices.tolist()] for maxs in value_index]
                    gold = [mapping[label] for label in b_labels.tolist()]
                    for ex in zip(b_ids, preds, gold):
                        dev_results.append(ex)
                else:
                    start_span, end_span = model(
                                        b_input_ids,
                                        attention_mask=b_attention_mask,
                                        token_type_ids=b_token_type_ids,
                                        srls = b_srls_idx,
                                        device=device,
                                        data_type=data_type
                                        )
                    start_value_index = [tensor.max(0) for tensor in start_span]
                    end_value_index = [tensor.max(0) for tensor in end_span]
                    start_acc = compute_acc([maxs.indices for maxs in start_value_index], b_labels.select(1, 0))
                    end_acc = compute_acc([maxs.indices for maxs in end_value_index], b_labels.select(1, 1))
                    acc = (start_acc + end_acc) / 2
                    start_loss = criterion(start_span, torch.unsqueeze(b_labels.select(1, 0), -1))
                    end_loss = criterion(end_span, torch.unsqueeze(b_labels.select(1, 1), -1))
                    loss = (start_loss + end_loss) / 2


            total_dev_loss += loss.item()
            total_dev_accuracy += acc

        avg_dev_accuracy = total_dev_accuracy / len(dev_idcs)
        print("  Development Accuracy: {0:.2f}".format(avg_dev_accuracy))
        avg_dev_loss = total_dev_loss / len(dev_data)
        dev_time = format_time(time.time() - t0)
        print("  Dev Loss: {0:.2f}".format(avg_dev_loss))
        print("  Dev took: {:}".format(dev_time))

        ################
        ### Test Run ###

        print("")
        print(green + "Running Test Set evaluation..." + end)

        t0 = time.time()

        test_results = []

        total_test_accuracy = 0
        total_test_loss = 0

        for idcs in test_idcs:
            batch = test_data[idcs[0]:idcs[1]]
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            b_srls = batch[4]
            b:ids = batch[5]
            b_srls_idx = convert_SRLs_to_tensor(model.srl_model.dictionary, b_srls, device)

            with torch.no_grad():
                if not data_type == "qa":
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_attention_mask,
                                token_type_ids=b_token_type_ids,
                                srls = b_srls_idx,
                                device=device,
                                data_type=data_type
                                )
                    value_index = [tensor.max(0) for tensor in outputs]
                    acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                    loss = criterion(outputs, b_labels)

                    preds = [mapping[maxs.indices.tolist()] for maxs in value_index]
                    gold = [mapping[label] for label in b_labels.tolist()]
                    for ex in zip(b_ids, preds, gold):
                        test_results.append(ex)
                else:
                    start_span, end_span = model(
                                        b_input_ids,
                                        attention_mask=b_attention_mask,
                                        token_type_ids=b_token_type_ids,
                                        srls = b_srls_idx,
                                        device=device,
                                        data_type=data_type
                                        )
                    start_value_index = [tensor.max(0) for tensor in start_span]
                    end_value_index = [tensor.max(0) for tensor in end_span]
                    start_acc = compute_acc([maxs.indices for maxs in start_value_index], b_labels.select(1, 0))
                    end_acc = compute_acc([maxs.indices for maxs in end_value_index], b_labels.select(1, 1))
                    acc = (start_acc + end_acc) / 2
                    start_loss = criterion(start_span, torch.unsqueeze(b_labels.select(1, 0), -1))
                    end_loss = criterion(end_span, torch.unsqueeze(b_labels.select(1, 1), -1))
                    loss = (start_loss + end_loss) / 2

            total_test_loss += loss.item()
            total_test_accuracy += acc

        avg_test_accuracy = total_test_accuracy / len(test_idcs)
        print("  Test Accuracy: {0:.2f}".format(avg_test_accuracy))
        avg_test_loss = total_test_loss / len(test_data)
        test_time = format_time(time.time() - t0)
        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Train Loss": avg_train_loss,
                "Dev Loss": avg_dev_loss,
                "Test Loss": avg_test_loss,
                "Dev Accur.": avg_dev_accuracy,
                "Test Accur.": avg_test_accuracy,
                "Train Time": train_time,
                "Dev Time": dev_time,
                "Test Time": test_time
            }
        )

        results.append(
            {
                epoch_i + 1: {
                    "dev": dev_results,
                    "test": test_results
                    }
            }
        )

        if config["early_stopping"]:
            if epoch_i > 0:
                if training_stats[-2]["Dev Loss"] < training_stats[-1]["Dev Loss"]:
                    if PATIENCE > 4:
                        print("")
                        print(red + "  !!! OVERFITTING !!!" + end)
                        print(red + "  Stopping fine-tuning!" + end)
                        break
                    PATIENCE += 1
                    print("")
                    print(
                        red + \
                        "Attention: Development loss increased for the {} time in series...".format(PATIENCE) + \
                        end
                        )
                else:
                    PATIENCE = 0

    if stats_file: write_stats(stats_file, training_stats, results)
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



def main():
    args = parse_cmd_args()
    global location
    location = args.location
    global data_set
    data_set = args.data_set
    global stats_file
    stats_file = args.stats_file
    config = load_json(args.config)
    fine_tune_BERT(config)
    


if __name__ == "__main__":
    main()
