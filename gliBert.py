import argparse
import datetime
import json
import time
import torch

import numpy as np

from load_data import dataloader
from torch import nn
from transformers import (
        AdamW,
        BertModel,
        BertTokenizer,
        get_linear_schedule_with_warmup
        )

merge_subs = MLQA_dataloader.merge_subs


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


# class SRL_Encoder(nn.Module):
#     def __init__(self, config):
#         super(SRL_Encoder, self).__init__()
#         self.config = config
#         self.embeddings = nn.Embedding(self.config["num_labels"], self.config["embedding_dim"])
#         self.encoder = nn.GRU(
#                             input_size=self.config["embedding_dim"],
#                             hidden_size=self.config["gru_hidden_size"],
#                             num_layers=self.config["num_layers"],
#                             bias=self.config["bias"],
#                             batch_first=True,
#                             dropout=self.config["gru_dropout"],
#                             bidirectional=self.config["bidirectional"]
#                             )
# 
#     def forward(self, tokens):
#         embeddings = self.embeddings(tokens)
#         output, _ = self.encoder(embeddings)
#         return output


class BertBinaryClassifier(nn.Module):
    def __init__(self, path, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens):
        _, pooled_output = self.bert(tokens)
        linear_output = self.linear(pooled_output)
        proba = self.sigmoid(linear_output)
        return proba


class BertEntailmentClassifierCLS(nn.Module):
    def __init__(self, config, num_classes, max_len):
        super(BertEntailmentClassifierCLS, self).__init__()
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


class GLIBert(nn.Module):
    def __init__(self, config, num_classes, max_len):
        super(GLIBert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config[location]["BERT"])
        self.tokenizer = BertTokenizer.from_pretrained(self.config[location]["BERT"])
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        if not SPAN_FLAG:
            self.head_layer = nn.Sequential(
                nn.Dropout(self.config["dropout"]),
                nn.Linear(max_len*768, self.config["head_hidden_size"]),
                nn.ReLU(inplace=True),
                nn.Linear(self.config["head_hidden_size"], num_classes),
            )
        else:
            self.start_span_layer = nn.Sequential(
                nn.Dropout(self.config["dropout"]),
                nn.Linear(max_len*768, self.config["head_hidden_size"]),
                nn.ReLU(inplace=True),
                nn.Linear(self.config["head_hidden_size"], max_len),
            )
            self.end_span_layer = nn.Sequential(
                nn.Dropout(self.config["dropout"]),
                nn.Linear(max_len*768, self.config["head_hidden_size"]),
                nn.ReLU(inplace=True),
                nn.Linear(self.config["head_hidden_size"], max_len),
            )
    
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

    def reconstruct_word_level(self, batch, ids):
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
        word_level_batch = []
        for j, sentence in enumerate(batch):
            word_level_sentence = []
            for i, token in enumerate(sentence):
                decode_token = self.tokenizer.decode([ids[j][i]])
                if decode_token.startswith("##"):
                    continue
                elif decode_token == "[PAD]":
                    word_level_sentence.append(token)
                    break
                elif i + 1 == len(sentence):
                    word_level_sentence.append(token)
                elif not self.tokenizer.decode([ids[j][i+1]]).startswith("##"):
                    word_level_sentence.append(token)
                else:
                    current_word = [token]
                    for k, subtoken in enumerate(sentence[i+1:]):
                        decode_subtoken = self.tokenizer.decode([ids[j][i+k+1]])
                        if decode_subtoken.startswith("##"):
                            current_word.append(subtoken)
                        else:
                            break
                    current_word = torch.stack(tuple(current_word))
                    mean_embs_word = torch.mean(current_word, 0)
                    word_level_sentence.append(mean_embs_word)
            pad_token = sentence[-1]
            while len(word_level_sentence) < len(sentence):
                word_level_sentence.append(pad_token)
            word_level_batch.append(torch.stack(tuple(word_level_sentence)))
        return_batch = torch.stack(tuple(word_level_batch))
        
        return return_batch
    
    def forward(
            self,
            tokens,
            attention_mask=None,
            token_type_ids=None,
            data_type=None,
            device=torch.device("cpu")
            ):
        if data_type == 2 or data_type == "qa":
            last_hidden_state, _ = self.bert(
                                        tokens,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids
                                        )
        else:
            last_hidden_state, _ = self.bert(
                                        tokens,
                                        attention_mask=attention_mask
                                        )

        if self.config["merge_subtokens"] == True:
            full_word_hidden_state = self.reconstruct_word_level(last_hidden_state, tokens) 
        reshaped_last_hidden = torch.reshape(
                full_word_hidden_state if self.config["merge_subtokens"] == True else last_hidden_state, 
                (
                    last_hidden_state.shape[0], 
                    last_hidden_state.shape[1]*last_hidden_state.shape[2])
                )
        if not SPAN_FLAG:
            output = self.head_layer(reshaped_last_hidden)
            #non_output = Swish(output)
            proba = self.softmax(output)
            return proba
        else:
            #output = self.span_layer(reshaped_last_hidden)
            start_span_output = self.start_span_layer(reshaped_last_hidden)
            end_span_output = self.end_span_layer(reshaped_last_hidden)
            start_span_proba = self.softmax(start_span_output)
            end_span_proba = self.softmax(end_span_output)
            new_end_span_proba = self.ensure_end_span_behind_start_span(
                                        start_span_proba,
                                        end_span_proba, 
                                        device
                                        )
            return start_span_proba, new_end_span_proba


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


def write_stats(stats_file, training_stats):
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


def print_preds(model, example, prediction, true_label, mapping, step, len_data, elapsed, merge):
    if not SPAN_FLAG:
        print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len_data, elapsed))
        print("  Last prediction: ")
        print("    Text:   {}".format(model.tokenizer.decode(example, skip_special_tokens=True)))
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

    # srl_encoder = SRL_Encoder(config)
    model = bert_head(config, num_classes, max_len)


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
    total_t0 = time.time()
    PATIENCE = 0

    for epoch_i in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")
        t0 = time.time()
        total_train_loss = 0

        #################
        ### Train Run ###

        model.train()
        for step, batch in enumerate(train_data):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)
            model.zero_grad()
            if not SPAN_FLAG:
                outputs = model(
                            b_input_ids, 
                            attention_mask=b_attention_mask,
                            token_type_ids=b_token_type_ids,
                            data_type=data_type,
                            device=device
                            )
                if step % print_stats == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    print_preds(
                            model,
                            b_input_ids[-1],
                            outputs[-1],
                            b_labels[-1],
                            mapping,
                            step,
                            len(dev_data),
                            elapsed,
                            merge_subtokens
                            )
                loss = criterion(outputs, b_labels)
            else:
                start_span, end_span = model(
                                        b_input_ids,
                                        attention_mask=b_attention_mask,
                                        token_type_ids=b_token_type_ids,
                                        data_type=data_type,
                                        device=device
                                        )
                if step % print_stats == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    print_preds(
                            model,
                            b_input_ids[-1],
                            (start_span[-1], end_span[-1]),
                            b_labels[-1],
                            mapping, step,
                            len(dev_data),
                            elapsed,
                            merge_subtokens
                            )
                start_loss = criterion(start_span, b_labels.select(1, 0))
                end_loss = criterion(end_span, b_labels.select(1, 1))
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
        print("Running Development Set evaluation...")
        t0 = time.time()
        model.eval()

        total_dev_accuracy = 0
        total_dev_loss = 0

        for batch in dev_data:
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)

            with torch.no_grad():
                if not SPAN_FLAG:
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_attention_mask,
                                token_type_ids=b_token_type_ids,
                                data_type=data_type,
                                device=device
                                )
                    value_index = [tensor.max(0) for tensor in outputs]
                    acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                    loss = criterion(outputs, b_labels)
                else:
                    start_span, end_span = model(
                                        b_input_ids,
                                        attention_mask=b_attention_mask,
                                        token_type_ids=b_token_type_ids,
                                        data_type=data_type,
                                        device=device
                                        )
                    start_value_index = [tensor.max(0) for tensor in start_span]
                    end_value_index = [tensor.max(0) for tensor in end_span]
                    start_acc = compute_acc([maxs.indices for maxs in start_value_index], b_labels.select(1, 0))
                    end_acc = compute_acc([maxs.indices for maxs in end_value_index], b_labels.select(1, 1))
                    acc = (start_acc + end_acc) / 2
                    start_loss = criterion(start_span, b_labels.select(1, 0))
                    end_loss = criterion(end_span, b_labels.select(1, 1))
                    loss = (start_loss + end_loss) / 2

            total_dev_loss += loss.item()
            total_dev_accuracy += acc

        avg_dev_accuracy = total_dev_accuracy / len(dev_data)
        print("  Development Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_dev_loss = total_dev_loss / len(dev_data)
        dev_time = format_time(time.time() - t0)
        print("  Dev Loss: {0:.2f}".format(avg_val_loss))
        print("  Dev took: {:}".format(validation_time))

        ################
        ### Test Run ###

        print("")
        print("Running Test Set evaluation...")

        t0 = time.time()

        total_test_accuracy = 0
        total_test_loss = 0

        for batch in test_data:
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)

            with torch.no_grad():
                if not SPAN_FLAG:
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_attention_mask,
                                token_type_ids=b_token_type_ids,
                                data_type=data_type,
                                device=device
                                )
                    value_index = [tensor.max(0) for tensor in outputs]
                    acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                    loss = criterion(outputs, b_labels)
                else:
                    start_span, end_span = model(
                                        b_input_ids,
                                        attention_mask=b_attention_mask,
                                        token_type_ids=b_token_type_ids,
                                        data_type=data_type,
                                        device=device
                                        )
                    start_value_index = [tensor.max(0) for tensor in start_span]
                    end_value_index = [tensor.max(0) for tensor in end_span]
                    start_acc = compute_acc([maxs.indices for maxs in start_value_index], b_labels.select(1, 0))
                    end_acc = compute_acc([maxs.indices for maxs in end_value_index], b_labels.select(1, 1))
                    acc = (start_acc + end_acc) / 2
                    start_loss = criterion(start_span, b_labels.select(1, 0))
                    end_loss = criterion(end_span, b_labels.select(1, 1))
                    loss = (start_loss + end_loss) / 2

            total_test_loss += loss.item()
            total_test_accuracy += acc

        avg_test_accuracy = total_test_accuracy / len(test_data)
        print("  Test Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_test_loss = total_test_loss / len(test_data)
        test_time = format_time(time.time() - t0)
        print("  Test Loss: {0:.2f}".format(avg_val_loss))
        print("  Test took: {:}".format(validation_time))

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Train Loss": avg_train_loss,
                "Dev Loss": avg_dev_loss,
                "Test Loss": avg_val_loss,
                "Dev Accur.": avg_dev_accuracy,
                "Test Accur.": avg_val_accuracy,
                "Train Time": train_time,
                "Dev Time": dev_time,
                "Test Time": test_time
            }
        )

        if config["early_stopping"]:
            if epoch_i > 1:
                if training_stats[-2]["Valid. Loss"] < training_stats[-1]["Valid. Loss"]:
                    if PATIENCE > 4:
                        print("")
                        print("  !!! OVERFITTING !!!")
                        print("  Stopping fine-tuning!")
                        break
                    PATIENCE += 1
                    print("")
                    print("Attention: Validation loss increased for the {} time in series...".format(PATIENCE))
                else:
                    PATIENCE = 0

    if stats_file: write_stats(stats_file, training_stats)
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



def main():
    args = parse_cmd_args()
    global location
    location = args.location
    global data_set
    data_set = args.data_set
    global SPAN_FLAG
    SPAN_FLAG = False if data_set not in ["MLQA", "XQuAD"] else True
    global stats_file
    stats_file = args.stats_file
    config = load_json(args.config)
    fine_tune_BERT(config)
    


if __name__ == "__main__":
    main()
