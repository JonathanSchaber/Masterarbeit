import argparse
import datetime
import json
import random
import time
import torch

import numpy as np

from load_data import Dataloader, dataloader
from torch import nn
from torch.optim import Adadelta
from transformers import (
        AdamW,
        BertModel,
        BertTokenizer,
        get_linear_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup
        )
from typing import List

merge_subs = Dataloader.merge_subs

blue = "\033[94m"

green = "\033[92m"

red = "\033[93m"

end = "\033[0m"


def parse_cmd_args():
    """Parse command line arguments.

    Returns:
        parser: argparse object
    """
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
            choices=["local", "remote"]
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


# def sigmoid(x):
#     return 1/(1+np.exp(-x))


# def swish(x):
#     return x * sigmoid(x)


# def Swish(batch):
#     swish_tensors = []
#     for tensor in batch:
#         swish_tensors.append(torch.tensor(list(map(swish, tensor))))
#     return torch.stack(tuple(batch))


class SRL_Encoder(nn.Module):
    """Implements RNN encoder for SRLs.

    Attributes:
        config: configuration file specifying various hyper params
        dictionary: mapping from SRLs to indices
        embeddings: embedding class from torch
        encoder: actual RNN module that computes embeddings (GRU)
    """

    def __init__(self, config):
        super(SRL_Encoder, self).__init__()
        self.config = config
        self.dictionary = {
            "B-A0":   	0,
            "B-A1":   	1,
            "B-A2":   	2,
            "B-A3":   	3,
            "B-A4":   	4,
            "B-A5":   	5,
            "B-A6":   	6,
            "B-A7":   	7,
            "B-A8":   	8,
            "B-A9":   	9,
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
            "B-V":   	22,
            "I-A0":  	23,
            "I-A1":  	24,
            "I-A2":  	25,
            "I-A3":  	26,
            "I-A4":  	27,
            "I-A5":  	28,
            "I-A6":  	29,
            "I-A7":  	30,
            "I-A8":  	31,
            "I-A9":  	32,
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
            "O":	    44,
            "0":        45,
            "[CLS]":    46,
            "[SEP]":    47
        }
        self.embeddings = nn.Embedding(len(set(self.dictionary.values())), self.config["SRL_embedding_dim"])
        #self.embeddings = nn.Embedding(len(self.dictionary), self.config["SRL_embedding_dim"])
        self.encoder = nn.GRU(
                            input_size=3*self.config["SRL_embedding_dim"],
                            hidden_size=self.config["SRL_hidden_size"],
                            num_layers=self.config["SRL_num_layers"],
                            bias=self.config["SRL_bias"],
                            batch_first=True,
                            dropout=self.config["SRL_dropout"],
                            bidirectional=self.config["SRL_bidirectional"]
                        )

    def convert_SRLs_to_tensor(
            self,
            lst: List[List[List[List[List[str]]]]],
            device: str="cpu") -> List[List[List[List["torch.Tensor"]]]]:
        """
        turns a nested list of SRLs into a list of tensors of indices of SRLs

        Args:
            lst: batch:AB:sentences:predicates:SRLs
            device: torch.device
        Returns:
            input lst with last list converted to torch.Tensor
        """
        new_lst = []
        for batch in lst:
            new_batch = []
            for AB in batch:
                new_AB = []
                for sentence in AB:
                    new_sentence = []
                    for predicate in sentence:
                        new_predicate = [self.dictionary[srl] for srl in predicate]
                        new_predicate = torch.tensor(new_predicate).to(device)
                        new_sentence.append(new_predicate)
                    new_AB.append(new_sentence)
                new_batch.append(new_AB)
            new_lst.append(new_batch)

        return new_lst

    def forward(self, tokens: List["torch.Tensor"]) -> List["torch.Tensor"]:
        """Forward pass - actual embedding

        Args:
            tokens: list of indices in torch tensors

        Returns:
            batch_outputs: list of torch tensor embeddings
        """
        batch_outputs = []
        for batch in tokens:
            sentences = []
            for sentence in batch:
                num_preds = len(sentence)
                pred_1 = self.embeddings(sentence[0])
                pred_2 = self.embeddings(sentence[1])
                pred_3 = self.embeddings(sentence[2])
                pred_merge = []
                for i in range(len(pred_1)):
                    toks = [pred_1[i], pred_2[i], pred_3[i]]
                    pred_merge.append(torch.cat(tuple(toks), dim=0))
                preds = torch.unsqueeze(torch.stack(pred_merge), dim=0)

                output, _ = self.encoder(preds)
                sentences.append(torch.squeeze(output, dim=0))
            sequence = torch.cat(tuple(sentences), dim=0)
            batch_outputs.append(sequence)

        return batch_outputs


class BertBase(nn.Module):
    """The Base model class from which all inherit
    """


    @staticmethod
    def _split_SRLs_to_subtokens(
            batch_srls: List[List[List["torch.Tensor"]]],
            batch_idxs: List[List[int]]) -> List[List[List["torch.Tensor"]]]:
        """Method for splitting SRLs to BERT subtokens

        Args:
            batch_srls: the SRLs
            batch_idxs: number indicating how many times element has to be split
        Returns:
            split_srls: splitted SRLs
        """
        split_srls = []

        for example in zip(batch_srls, batch_idxs):
            split_example = []
            offset = 0
            sentence_idxs = []
            for sent in example[0]:
                sentence_idxs.append(example[1][offset:offset+len(sent[0])])
                offset += len(sent[0])
            sentence_idxs = [sentence_idx for sentence_idx in sentence_idxs if not len(sentence_idx) == 0]
            for sentence in zip(example[0], sentence_idxs):
                split_sentence = []
                for predicate in sentence[0]:
                    new_predicate = []
                    for i, srl in enumerate(predicate[:len(sentence[1])]):
                        copies = [srl]*sentence[1][i]
                        for copy in copies:
                            new_predicate.append(copy)
                    assert len(new_predicate) == sum(sentence[1])
                    split_sentence.append(torch.stack(new_predicate))
                split_example.append(split_sentence)
            split_srls.append(split_example)

        #import ipdb; ipdb.set_trace()
        return split_srls

    @staticmethod
    def _get_AB_SRLs(lst):
        a_lst, b_lst = ([] for i in range(2))
        for batch in lst:
            a_lst.append(batch[0])
            b_lst.append(batch[1])
        return a_lst, b_lst

    @staticmethod
    def _get_A_SRLs(lst):
        a_lst = []
        for batch in lst:
            a_lst.append(batch[0])
        return a_lst

    @staticmethod
    def _concatenate_sent1_sent2(sent1, sent2):
        new_batch = []
        for batch in zip(sent1, sent2):
            concat = []
            concat.append(torch.cat((batch[0][0][0], batch[1][0][0]), dim=0))
            concat.append(torch.cat((batch[0][0][1], batch[1][0][1]), dim=0))
            concat.append(torch.cat((batch[0][0][2], batch[1][0][2]), dim=0))
            new_batch.append([concat])

        return new_batch

    def _concatenate_sents(self, srls):
        """concatenates the sub-sentences of a seq

        Two ways of doing this, controlled by config-param «zeros»:
        If there are less than 3 predicates per subsentence, either
        (1) append first one as many times as needed, or (2) add
        SRL 0 (no SRL info).

        Args:
            srls: batch of SRLs
        Returns:
            new_batch: concatenated SRLs (first 3 preds)
        """
        new_batch = []
        for batch in srls:
            concat = []
            concat.append(torch.cat([sent[0] for sent in batch], dim=0))
            if not self.config["zeros"]:
                # if less than 3 preds, duplicate first SRLs
                concat.append(torch.cat([sent[1] if len(sent) > 1 else sent[0] for sent in batch], dim=0))
                concat.append(torch.cat([sent[2] if len(sent) > 2 else sent[0] for sent in batch], dim=0))
            else:
                # if less than 3 preds, simply add "0" SRLs
                concat.append(torch.cat([sent[1]
                                            if len(sent) > 1
                                            else torch.squeeze(
                                                    torch.stack([self.zero_srl]*len(sent[0])), dim=-1)
                                            for sent
                                            in batch], dim=0))
                concat.append(torch.cat([sent[2]
                                            if len(sent) > 2
                                            else torch.squeeze(
                                                    torch.stack([self.zero_srl]*len(sent[0])), dim=-1)
                                            for sent
                                            in batch], dim=0))
            new_batch.append([concat])

        return new_batch

    def _add_spec_srl(self, srls, spec):
        """adds «meta» SRL at beginning of seq

        Args:
            srls: batch of SRLs
            spec: defines which meta-SRL
        Return:
            batch of SRLs with addded meta SRL
        """
        new_batch = []
        if spec == "[CLS]":
            spec_srl = self.cls_srl
        elif spec == "[SEP]":
            spec_srl = self.sep_srl
        for batch in srls:
            new_sentence = []
            for i, sentence in enumerate(batch):
                new_predicate = []
                for predicate in sentence:
                    if i == 0:
                        new_pred = torch.cat((spec_srl, predicate), dim=-1)
                        new_predicate.append(new_pred)
                    else:
                        new_predicate.append(predicate)
                new_sentence.append(new_predicate)
            new_batch.append(new_sentence)

        return new_batch

    def embed_srls(self, srls, split_idxs, data_type):
        """embeds a batch of SRLs

        Args:
            srls: batch of SRL-idxs
            split_idxs: information which SRLs need to be split
            data_type: 1-sent or 2-sent data
        Returns:
            emb: embedded batch
        """
        if data_type != 1:
            a_srls, b_srls = self._get_AB_SRLs(srls)
            if not self.config["merge_subtokens"]:
                a_srls, b_srls = (self._split_SRLs_to_subtokens(a_srls, split_idxs[0]),
                                 self._split_SRLs_to_subtokens(b_srls, split_idxs[1]))
            one_a = self._concatenate_sents(a_srls)
            one_a = self._add_spec_srl(one_a, "[CLS]")
            one_b = self._concatenate_sents(b_srls)
            one_b = self._add_spec_srl(one_b, "[SEP]")
            one_sent = self._concatenate_sent1_sent2(one_a, one_b)
        else:
            srls = self._get_A_SRLs(srls)
            if not self.config["merge_subtokens"]:
                srls = self._split_SRLs_to_subtokens(srls, split_idxs[0])
            one_sent = self._concatenate_sents(srls)
            one_sent = self._add_spec_srl(one_sent, "[CLS]")
        emb = self.srl_model(one_sent)

        return emb

    def create_dummy_srl(self, device):
        # *2 because bi-GRU
        dummy = torch.unsqueeze(torch.tensor([0.0]*2*self.config["SRL_hidden_size"]), dim=0)
        self.dummy_srl = dummy.to(device)

    def create_cls_srl(self, device):
        cls_tensor = torch.unsqueeze(torch.tensor(self.srl_model.dictionary["[CLS]"]), dim=0)
        self.cls_srl = cls_tensor.to(device)

    def create_sep_srl(self, device):
        sep_tensor = torch.unsqueeze(torch.tensor(self.srl_model.dictionary["[SEP]"]), dim=0)
        self.sep_srl = sep_tensor.to(device)

    def create_zero_srl(self, device):
        zero_tensor = torch.unsqueeze(torch.tensor(self.srl_model.dictionary["0"]), dim=0)
        self.zero_srl = zero_tensor.to(device)

    def pad_SRLs(self, batch, dummy, length=None):
        """Pad a batch of token SRLs to tokenized batch

        Args:
            batch: the batch list of SRLs
            dummy: the dummy-SRL (all 0.0)
            length: only used when something cut of ([CLS] sometimes e.g.)
        Returns:
            new_batch: same dims as input
        """
        length = self.max_len if length == None else length
        new_batch = []
        for example in batch:
            if not len(example) <= length:
                #import ipdb; ipdb.set_trace()
                example = example[:length][:]
            lst = [example] + [dummy]*(length-len(example))
            tensor = torch.cat(tuple(lst), dim=0)
            new_batch.append(tensor)

        new_batch = torch.stack(new_batch)

        return new_batch

    def reconstruct_word_level(self, batch, ids, device="cpu"):
        """Method for joining subtokenized words back to word level.
        The strategy is to average subtoken embeddings.
        To preserve original length, append non-informative zero vector until
        original length of sequence is reached.

        Args:
            batch: torch.tensor embeddings of subtokens
            ids: torch.tensor indices of subtokens
            decive: CPU/GPU
        Returns:
            return_batch: torch.tensor of same input dimensions with averaged embeddings
                        for subtokens
            ab_batch_idx: split idx used when contrary needs to be done: SLRs splitting
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
                elif decode_token == "[CLS]":
                    word_level_sentence.append(token)
                elif decode_token in ["[MASK]", "[UNK]"]:
                    word_level_sentence.append(token)
                    first_second.append(1)
                elif decode_token == "[SEP]":
                    first_second = B_sent_idx
                    if i + 1 == len(sentence):
                        break
                elif decode_token == "[PAD]":
                    break
                elif not self.tokenizer.decode([ids[j][i+1]]).startswith("##"):
                    word_level_sentence.append(token)
                    first_second.append(1)
                else:
                    current_word = [token]
                    split_counter = 1
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
            # pad_token = sentence[-1]
            while len(word_level_sentence) < len(sentence):
                word_level_sentence.append(pad_token)
            word_level_batch.append(torch.stack(tuple(word_level_sentence)))
            batch_idx.append((A_sent_idx, B_sent_idx))
            ab_batch_idx = [[idx[0] for idx in batch_idx], [idx[1] for idx in batch_idx]]

        return_batch = torch.stack(tuple(word_level_batch))

        return return_batch, ab_batch_idx


class GliBertClassifierCLS(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        if config["combine_SRLs"]:
            self.linear = nn.Linear((768+2*config["SRL_hidden_size"]), num_classes)
        else:
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
        last_hidden_state, pooler_output = self.bert(
                                tokens,
                                attention_mask,
                                token_type_ids
                                )
        _, split_idxs = self.reconstruct_word_level(last_hidden_state,
                                        tokens,
                                        device)

        if self.config["combine_SRLs"]:
            emb = self.embed_srls(srls, split_idxs, data_type)
            emb = torch.stack([batch[0,:] for batch in emb])

            pooler_output = torch.cat((pooler_output, emb), dim=-1)

        linear_output = self.linear(pooler_output)
        proba = self.softmax(linear_output)

        return proba


class GliBertClassifierFFNN(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.dummy_srl = None
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        if config["combine_SRLs"]:
            self.linear = nn.Linear((768+2*config["SRL_hidden_size"])*max_len, num_classes)
        else:
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
        last_hidden_state, _ = self.bert(tokens,
                                        attention_mask,
                                        token_type_ids)
        full_word_hidden_state, split_idxs = self.reconstruct_word_level(last_hidden_state,
                                        tokens,
                                        device)
        hidden_state = full_word_hidden_state if self.config["merge_subtokens"] else last_hidden_state

        if self.config["combine_SRLs"]:
            srl_emb = self.embed_srls(srls, split_idxs, data_type)

            srl_batch = self.pad_SRLs(srl_emb, self.dummy_srl)
            combo_merge_batch = torch.cat((hidden_state, srl_batch), dim=-1)

            reshaped_last_hidden = torch.reshape(
                    combo_merge_batch,
                    (
                        combo_merge_batch.shape[0],
                        combo_merge_batch.shape[1]*combo_merge_batch.shape[2])
                    )
        else:
            reshaped_last_hidden = torch.reshape(
                    hidden_state,
                    (
                        hidden_state.shape[0],
                        hidden_state.shape[1]*hidden_state.shape[2])
                    )
        linear_output = self.linear(reshaped_last_hidden)
        proba = self.softmax(linear_output)

        return proba


class GliBertClassifierFFNNNoCLS(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.dummy_srl = None
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        if config["combine_SRLs"]:
            self.linear = nn.Linear((768+2*config["SRL_hidden_size"])*(max_len-1), num_classes)
        else:
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
        last_hidden_state, _ = self.bert(tokens,
                                        attention_mask,
                                        token_type_ids)
        full_word_hidden_state, split_idxs = self.reconstruct_word_level(last_hidden_state,
                                        tokens,
                                        device)
        hidden_state = full_word_hidden_state if self.config["merge_subtokens"] else last_hidden_state
        hidden_state = hidden_state[:, 1:, :]

        if self.config["combine_SRLs"]:
            srl_emb = self.embed_srls(srls, split_idxs, data_type)
            srl_batch = self.pad_SRLs(srl_emb, self.dummy_srl, self.max_len-1)
            combo_merge_batch = torch.cat((hidden_state, srl_batch), dim=-1)

            reshaped_last_hidden = torch.reshape(
                    combo_merge_batch,
                    (
                        combo_merge_batch.shape[0],
                        combo_merge_batch.shape[1]*combo_merge_batch.shape[2])
                    )
        else:
            reshaped_last_hidden = torch.reshape(
                    hidden_state,
                    (
                        hidden_state.shape[0],
                        hidden_state.shape[1]*hidden_state.shape[2])
                    )
        linear_output = self.linear(reshaped_last_hidden)
        proba = self.softmax(linear_output)

        return proba


class GliBertClassifierGRU(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.dummy_srl = None
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        self.gru = nn.GRU(
                input_size=768+2*config["SRL_hidden_size"] if config["combine_SRLs"] else 768,
                hidden_size=config["GRU_head_hidden_size"],
                num_layers=2,
                bias=True,
                batch_first=True,
                dropout=config["GRU_head_dropout"],
                bidirectional=True
                )
        self.linear = nn.Linear(2*config["GRU_head_hidden_size"], num_classes)
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
        last_hidden_state, _ = self.bert(tokens,
                                        attention_mask,
                                        token_type_ids)
        full_word_hidden_state, split_idxs = self.reconstruct_word_level(last_hidden_state,
                                        tokens,
                                        device)
        hidden_state = full_word_hidden_state if self.config["merge_subtokens"] else last_hidden_state

        if self.config["combine_SRLs"]:
            srl_emb = self.embed_srls(srls, split_idxs, data_type)

            srl_batch = self.pad_SRLs(srl_emb, self.dummy_srl)
            combo_merge_batch = torch.cat((hidden_state, srl_batch), dim=-1)

            _, h_n = self.gru(combo_merge_batch)
        else:
            _, h_n = self.gru(hidden_state)
        hidden = h_n.view(2, 2, tokens.shape[0], self.config["GRU_head_hidden_size"])
        last_hidden = hidden[-1]
        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]
        comb = torch.cat((last_hidden_fwd, last_hidden_bwd), dim=1)
        linear_output = self.linear(comb)
        proba = self.softmax(linear_output)

        return proba


class GliBertClassifierCNN(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.dummy_srl = None
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        self.cnn = nn.Sequential(
                    nn.Conv1d(
                        768 if not config["combine_SRLs"] else 768+2*config["SRL_hidden_size"],
                        1,
                        kernel_size=10,
                        stride=1,
                        padding=10
                        ),
                    nn.Tanh(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
        self.linear = nn.Linear(105, num_classes)
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
        last_hidden_state, _ = self.bert(tokens,
                                        attention_mask,
                                        token_type_ids)
        full_word_hidden_state, split_idxs = self.reconstruct_word_level(last_hidden_state,
                                        tokens,
                                        device)
        hidden_state = full_word_hidden_state if self.config["merge_subtokens"] else last_hidden_state

        if self.config["combine_SRLs"]:
            srl_emb = self.embed_srls(srls, split_idxs, data_type)

            srl_batch = self.pad_SRLs(srl_emb, self.dummy_srl)

            combo_merge_batch = torch.cat((hidden_state, srl_batch), dim=-1)

            # swap dimensions, so embedding-dimension are input channels
            combo_merge_batch = combo_merge_batch.permute(0, 2, 1)

            cnn_output = self.cnn(combo_merge_batch)
        else:
            # swap dimensions, so embedding-dimension are input channels
            hidden_state = hidden_state.permute(0, 2, 1)
            cnn_output = self.cnn(hidden_state)
        linear_output = self.linear(cnn_output)
        proba = self.softmax(linear_output)
        proba = torch.stack([torch.squeeze(tensor) for tensor in proba])

        return proba


class GliBertSpanPrediction(BertBase):
    def __init__(self, config, num_classes, max_len):
        super(BertBase, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.srl_model = SRL_Encoder(config)
        self.dummy_srl = None
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        if config["combine_SRLs"]:
            self.linear = nn.Linear(768+2*config["SRL_hidden_size"], 2)
        else:
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
        full_word_hidden_state, split_idxs = self.reconstruct_word_level(last_hidden_state,
                                        tokens,
                                        device)
        hidden_state = full_word_hidden_state if self.config["merge_subtokens"] else last_hidden_state

        if self.config["combine_SRLs"]:
            srl_emb = self.embed_srls(srls, split_idxs, data_type)

            srl_batch = self.pad_SRLs(srl_emb, self.dummy_srl)
            combo_merge_batch = torch.cat((hidden_state, srl_batch), dim=-1)

            linear_output = self.linear(combo_merge_batch)
        else:
            linear_output = self.linear(hidden_state)
        start_logits, end_logits = linear_output.split(1, dim=-1)
        start_span = self.softmax(start_logits)
        end_span = self.softmax(end_logits)

        return start_span, end_span


def write_stats(stats_file, training_stats, training_results):
    """checks if outfile specified, if yes, writes stats to it

    Args:
        stats_file: filepath to stats_file
        training_stats: statistics of training
        training_results: actual predictions
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
        print("Statistics written to file: {}".format(stats_file))

        results_file = stats_file.rstrip(".json") + ".results.json"
        with open(results_file, "w") as outfile:
            outfile.write(json.dumps(training_results))
        print("Results written to file: {}".format(results_file))


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def compute_acc(preds, labels):
    """computes the accordance of two lists

    Args:
        preds: list
        labels: list
    Returns:
        accuracy
    """
    correct = 0
    assert len(preds) == len(labels)
    for pred, lab in zip(preds, labels):
        if pred == lab: correct += 1

    return correct / len(preds)


def print_preds(model,
                data_type,
                example,
                srls,
                prediction,
                true_label,
                mapping,
                step,
                len_data,
                elapsed, merge):
    reverse_dict = {value: key for key, value in model.srl_model.dictionary.items()}
    first_srls = [predicate[0].tolist() for sentence in srls for predicate in sentence]
    first_srls = [reverse_dict[srl] for ls in first_srls for srl in ls]
    second_srls = [predicate[1].tolist() if len(predicate) > 1 else [45]*len(predicate[0]) for sentence in srls for predicate in sentence]
    second_srls = [reverse_dict[srl] for ls in second_srls for srl in ls]
    third_srls = [predicate[2].tolist() if len(predicate) > 2 else [45]*len(predicate[0]) for sentence in srls for predicate in sentence]
    third_srls = [reverse_dict[srl] for ls in third_srls for srl in ls]
    tokens = model.tokenizer.tokenize(model.tokenizer.decode(example))
    tokens = merge_subs(tokens)
    tokens = [tok for tok in tokens if tok not in ["[CLS]", "[SEP]", "[PAD]"]]
    if not data_type == "qa":
        prediction = prediction[-1]
        print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len_data, elapsed))
        print("  Last example: ")
        # if not len(tokens) == len(first_srls):
        #    import ipdb; ipdb.set_trace()
        for elem in zip(tokens, first_srls, second_srls, third_srls):
            print_tokens = []
            for token in elem:
                token = token + "\t\t" if len(token) < 8 else token + "\t"
                print_tokens.append(token)
            print("".join(print_tokens))
        print("    Prediction:  {}".format(mapping[prediction.max(0).indices.item()]))
        print("    True Label:  {}".format(mapping[true_label.item()]))
        print("")
    else:
        start_span, end_span = prediction
        start_span, end_span = start_span[-1], end_span[-1]
        ex = model.tokenizer.decode(example)
        sep_index = ex.index("[SEP]")
        question = ex[:sep_index].replace("[CLS] ", "")
        context = ex[sep_index:].replace("[SEP] ", "").replace("[PAD]", "").strip()
        print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len_data, elapsed))
        print("  Last example: ")
        print("    Context:   {}".format(context))
        print("    Question:   {}".format(question))
        if not merge:
            sentence = model.tokenizer.tokenize(
                                model.tokenizer.decode(example)
                                )
        else:
            sentence = merge_subs(model.tokenizer.tokenize(
                                model.tokenizer.decode(example)
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

    random.shuffle(batch_idcs)

    return batch_idcs


def fine_tune_BERT(config):
    """Define fine-tuning procedure, write results to file.

    Args:
        config: dictionary specifying hyper params
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

    global model
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

    model.create_dummy_srl(device)
    model.create_cls_srl(device)
    model.create_sep_srl(device)
    model.create_zero_srl(device)

    optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            eps=1e-8
        )

    total_steps = len(train_idcs) * epochs
    scheduler = get_linear_schedule_with_warmup(
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    training_stats = []
    training_stats.append({"data set": data_set})
    training_stats.append(config)
    results = []
    total_t0 = time.time()
    patience = 0

    for epoch_i in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print(blue + "Training..." + end)
        t0 = time.time()
        total_train_loss = 0
        total_train_accuracy = 0

        #################
        ### Train Run ###

        model.train()
        for step, idcs in enumerate(train_idcs):
            batch = train_data[idcs[0]:idcs[1]]
            global b_input_ids
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            b_srls = batch[4]
            b_srls_idx = model.srl_model.convert_SRLs_to_tensor(b_srls, device)
            model.zero_grad()


            outputs = model(
                        b_input_ids,
                        attention_mask=b_attention_mask,
                        token_type_ids=b_token_type_ids,
                        srls = b_srls_idx,
                        device=device,
                        data_type=data_type
                        )
            # add #epoch to print condition for some variation in print output
            if (step + epoch_i) % print_stats == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print_preds(
                        model,
                        data_type,
                        b_input_ids[-1],
                        b_srls_idx[-1],
                        outputs,
                        b_labels[-1],
                        mapping,
                        step,
                        len(train_idcs),
                        elapsed,
                        merge_subtokens
                        )

            if not data_type == "qa":
                value_index = [tensor.max(0) for tensor in outputs]
                acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                loss = criterion(outputs, b_labels)
            else:
                start_span, end_span = outputs
                start_value_index = [tensor.max(0) for tensor in start_span]
                end_value_index = [tensor.max(0) for tensor in end_span]
                start_acc = compute_acc([maxs.indices for maxs in start_value_index], b_labels.select(1, 0))
                end_acc = compute_acc([maxs.indices for maxs in end_value_index], b_labels.select(1, 1))
                acc = (start_acc + end_acc) / 2
                start_loss = criterion(start_span, torch.unsqueeze(b_labels.select(1, 0), -1))
                end_loss = criterion(end_span, torch.unsqueeze(b_labels.select(1, 1), -1))
                loss = (start_loss + end_loss) / 2

            # if step == 500:
            #     print("")
            #     print(red + "  >> Starting evaluating, train set is massive..." + end)
            #     print("")
            #     break

            total_train_accuracy += acc
            total_train_loss += loss.item()
            loss.backward()
            # This is to help prevent the "exploding gradients" problem. (Maybe not necessary?)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_accuracy = total_train_accuracy / (step+1)
        avg_train_loss = total_train_loss / (step+1)
        train_time = format_time(time.time() - t0)

        print("")
        print("  Average Train Accuracy: {0:.2f}".format(avg_train_accuracy))
        print("  Average Train Loss: {0:.2f}".format(avg_train_loss))
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
        all_preds = []
        all_labels = []
        all_pred_starts = []
        all_pred_ends = []
        all_label_starts = []
        all_label_ends = []

        for idcs in dev_idcs:
            batch = dev_data[idcs[0]:idcs[1]]
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            b_srls = batch[4]
            b_ids = batch[5]
            b_srls_idx = model.srl_model.convert_SRLs_to_tensor(b_srls, device)


            with torch.no_grad():
                outputs = model(
                            b_input_ids,
                            attention_mask=b_attention_mask,
                            token_type_ids=b_token_type_ids,
                            srls = b_srls_idx,
                            device=device,
                            data_type=data_type
                            )
                if not data_type == "qa":
                    value_index = [tensor.max(0) for tensor in outputs]
                    all_preds.append([maxs.indices for maxs in value_index])
                    all_labels.append(b_labels)
                    #acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                    loss = criterion(outputs, b_labels)

                    preds = [mapping[maxs.indices.tolist()] for maxs in value_index]
                    gold = [mapping[label] for label in b_labels.tolist()]
                    for ex in zip(b_ids, preds, gold):
                        dev_results.append(ex)
                else:
                    start_span, end_span = outputs
                    start_value_index = [tensor.max(0) for tensor in start_span]
                    end_value_index = [tensor.max(0) for tensor in end_span]
                    all_pred_starts.append([maxs.indices for maxs in start_value_index])
                    all_pred_ends.append([maxs.indices for maxs in end_value_index])
                    all_label_starts.append(b_labels.select(1, 0))
                    all_label_ends.append(b_labels.select(1, 1))
                    start_loss = criterion(start_span, torch.unsqueeze(b_labels.select(1, 0), -1))
                    end_loss = criterion(end_span, torch.unsqueeze(b_labels.select(1, 1), -1))
                    loss = (start_loss + end_loss) / 2

                    preds = zip([x.indices.item() for x in start_value_index],
                                [x.indices.item() for x in end_value_index])
                    gold = [tuple(x.tolist()) for x in b_labels]
                    for ex in zip(b_ids, preds, gold):
                        dev_results.append(ex)


            total_dev_loss += loss.item()
            #total_dev_accuracy += acc

        flatten = lambda ls: [item for batch in ls for item in batch]
        if data_type == "qa":
            start_acc = compute_acc(flatten(all_pred_starts), flatten(all_label_starts))
            end_acc = compute_acc(flatten(all_pred_ends), flatten(all_label_ends))
            avg_dev_accuracy = (start_acc + end_acc) / 2
        else:
            avg_dev_accuracy = compute_acc(flatten(all_preds), flatten(all_labels))
        #avg_dev_accuracy = total_dev_accuracy / len(dev_idcs)
        avg_dev_loss = total_dev_loss / len(dev_idcs)
        dev_time = format_time(time.time() - t0)

        print("")
        print("  Dev Accuracy: {0:.2f}".format(avg_dev_accuracy))
        print("  Average Dev Loss: {0:.2f}".format(avg_dev_loss))
        print("  Dev epoch took: {:}".format(dev_time))

        ################
        ### Test Run ###

        print("")
        print(green + "Running Test Set evaluation..." + end)

        t0 = time.time()

        test_results = []

        total_test_accuracy = 0
        total_test_loss = 0
        all_preds = []
        all_labels = []
        all_pred_starts = []
        all_pred_ends = []
        all_label_starts = []
        all_label_ends = []

        for idcs in test_idcs:
            batch = test_data[idcs[0]:idcs[1]]
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            b_srls = batch[4]
            b_ids = batch[5]
            b_srls_idx = model.srl_model.convert_SRLs_to_tensor(b_srls, device)

            with torch.no_grad():
                outputs = model(
                            b_input_ids,
                            attention_mask=b_attention_mask,
                            token_type_ids=b_token_type_ids,
                            srls = b_srls_idx,
                            device=device,
                            data_type=data_type
                            )
                if not data_type == "qa":
                    value_index = [tensor.max(0) for tensor in outputs]
                    all_preds.append([maxs.indices for maxs in value_index])
                    all_labels.append(b_labels)
                    #acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                    loss = criterion(outputs, b_labels)

                    preds = [mapping[maxs.indices.tolist()] for maxs in value_index]
                    gold = [mapping[label] for label in b_labels.tolist()]
                    for ex in zip(b_ids, preds, gold):
                        test_results.append(ex)
                else:
                    start_span, end_span = outputs
                    start_value_index = [tensor.max(0) for tensor in start_span]
                    end_value_index = [tensor.max(0) for tensor in end_span]
                    all_pred_starts.append([maxs.indices for maxs in start_value_index])
                    all_pred_ends.append([maxs.indices for maxs in end_value_index])
                    all_label_starts.append(b_labels.select(1, 0))
                    all_label_ends.append(b_labels.select(1, 1))
                    start_loss = criterion(start_span, torch.unsqueeze(b_labels.select(1, 0), -1))
                    end_loss = criterion(end_span, torch.unsqueeze(b_labels.select(1, 1), -1))
                    loss = (start_loss + end_loss) / 2

                    preds = zip([x.indices.item() for x in start_value_index],
                                [x.indices.item() for x in end_value_index])
                    gold = [tuple(x.tolist()) for x in b_labels]
                    for ex in zip(b_ids, preds, gold):
                        test_results.append(ex)

            total_test_loss += loss.item()
            #total_test_accuracy += acc

        flatten = lambda ls: [item for batch in ls for item in batch]
        if data_type == "qa":
            start_acc = compute_acc(flatten(all_pred_starts), flatten(all_label_starts))
            end_acc = compute_acc(flatten(all_pred_ends), flatten(all_label_ends))
            avg_test_accuracy = (start_acc + end_acc) / 2
        else:
            avg_test_accuracy = compute_acc(flatten(all_preds), flatten(all_labels))
        #avg_test_accuracy = total_test_accuracy / len(test_idcs)
        avg_test_loss = total_test_loss / len(test_idcs)
        test_time = format_time(time.time() - t0)

        print("")
        print("  Test Accuracy: {0:.2f}".format(avg_test_accuracy))
        print("  Average Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test epoch took: {:}".format(test_time))

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Train Loss": avg_train_loss,
                "Dev Loss": avg_dev_loss,
                "Test Loss": avg_test_loss,
                "Train Accur.": avg_train_accuracy,
                "Dev Accur.": avg_dev_accuracy,
                "Test Accur.": avg_test_accuracy,
                "Train Time": train_time,
                "Dev Time": dev_time,
                "Test Time": test_time,
                "patience": patience
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
                if training_stats[-2]["Dev Loss"] <= training_stats[-1]["Dev Loss"]:
                    if patience > 4:
                        print("")
                        print(red + "  !!! OVERFITTING !!!" + end)
                        print(red + "  Stopping fine-tuning!" + end)
                        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
                        break
                    patience += 1
                    print("")
                    print(
                        red +
                        "Attention: Development loss increased for the {} time in series...".format(patience) +
                        end
                        )
                else:
                    patience = 0

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    if stats_file:
        write_stats(stats_file, training_stats, results)


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

