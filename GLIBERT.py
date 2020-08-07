import time
import datetime
import json
import argparse
import torch

import numpy as np

#from datetime import datetime

from torch import nn
from torch.utils.data import (
        TensorDataset,
        random_split,
        DataLoader,
        RandomSampler,
        SequentialSampler
        )
from transformers import (
        BertTokenizer,
        BertModel,
        BertForQuestionAnswering,
        AdamW,
        get_linear_schedule_with_warmup
        )

from load_data import *


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
            choices=["XNLI", "SCARE", "PAWS-X"]
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
    def __init__(self, config, num_classes, max_len, dropout=0.1):
        super(BertEntailmentClassifierCLS, self).__init__()
        self.bert = BertModel.from_pretrained(config[location]["BERT"])
        self.tokenizer = BertTokenizer.from_pretrained(config[location]["BERT"])
        self.dense_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            #nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            #nn.Linear(768, num_classes),
        )
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, tokens):
        _, pooler_output = self.bert(tokens)
        linear_output = self.dense_layer(pooler_output)
        #non_linear_output = Swish(linear_output)
        proba = self.softmax(linear_output)
        return proba


class BertEntailmentClassifierAllHidden(nn.Module):
    def __init__(self, config, num_classes, max_len, dropout=0.1):
        super(BertEntailmentClassifierAllHidden, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config[location]["BERT"])
        self.tokenizer = BertTokenizer.from_pretrained(self.config[location]["BERT"])
        self.dense_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(max_len*768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
            #nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            #nn.Linear(768, num_classes),
        )
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def reconstruct_word_level(self, batch, ids):
        """method for joining subtokenized words back to word level
        The idea is to average subtoken embeddings.
        To preserve original length, append last subtoken-embedding (which
        is per definition [PAD] since padding + 1 of max- length) until original
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
    
    def forward(self, tokens):
        last_hidden_state, _ = self.bert(tokens)
        if self.config["merge_subtokens"] == True:
            full_word_hidden_state = self.reconstruct_word_level(last_hidden_state, tokens) 
        reshaped_last_hidden = torch.reshape(
                full_word_hidden_state if self.config["merge_subtokens"] == True else last_hidden_state, 
                (
                    last_hidden_state.shape[0], 
                    last_hidden_state.shape[1]*last_hidden_state.shape[2])
                )
        linear_output = self.dense_layer(reshaped_last_hidden)
        #non_linear_output = Swish(linear_output)
        proba = self.softmax(linear_output)
        return proba


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


def fine_tune_BERT(config, stats_file=None):
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

    train_data, test_data, num_classes, max_len, mapping = dataloader(config, location, data_set)
    model = bert_head(config, num_classes, max_len)
    mapping = {value: key for (key, value) in mapping.items()}


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

    total_steps = len(train_data) * epochs
    scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = 0,
            num_training_steps = total_steps
        )
    training_stats = []
    training_stats.append(config)
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_data):
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            model.zero_grad()
            outputs = model(b_input_ids)
            if step % print_stats == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(train_data), elapsed))
                print("  Last prediction: ")
                print("    Text:   {}".format(model.tokenizer.decode(b_input_ids[-1], skip_special_tokens=True)))
                print("    Prediction:  {}".format(mapping[outputs[-1].max(0).indices.item()]))
                print("    True Label:  {}".format(mapping[b_labels[-1].item()]))
                print("")

            loss = criterion(outputs, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            # This is to help prevent the "exploding gradients" problem. (Maybe not necessary?)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_data)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in test_data:
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids)
                value_index = [tensor.max(0) for tensor in outputs]
                acc = compute_acc([maxs.indices for maxs in value_index], b_labels)
                loss = criterion(outputs, b_labels)
            total_eval_loss += loss.item()
            total_eval_accuracy += acc

        avg_val_accuracy = total_eval_accuracy / len(test_data)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(test_data)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time
            }
        )

        if avg_train_loss < avg_val_loss:
            print("")
            print("  OVERFITTING: train loss: {:.3f}, validation loss: {:.3f}".format(
                                                                                avg_train_loss,
                                                                                avg_val_loss
                                                                                ))
            print("  Stopping fine-tuning!")
            break

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
    stats_file = args.stats_file
    config = load_json(args.config)
    fine_tune_BERT(config, stats_file)
    


if __name__ == "__main__":
    main()
