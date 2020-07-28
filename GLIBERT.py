import time
import datetime
import json
import argparse
import torch

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

from load_data import dataloader_XNLI


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--location', type=str, help='"local", "midgard" or "rattle". '
                                                           'Indicate which paths will be used.')
    parser.add_argument("-c", "--config", type=str, help="Path to hyperparamter/config file (json).")
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, "r") as f:
        f = f.read()
        data = json.loads(f)
    return data


class BertBinaryClassifier(nn.Module):
    def __init__(self, path, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens):
        _, pooled_output = self.bert(tokens)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba


class BertEntailmentClassifier(nn.Module):
    def __init__(self, path, out_classes=3, dropout=0.1):
        super(BertEntailmentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.linear = nn.Linear(768, out_classes)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, tokens):
        _, pooled_output = self.bert(tokens)
        linear_output = self.linear(dropout_output)
        proba = self.softmax(linear_output)
        return proba


def create_model_and_tokenizer(path_to_model):
    """Create respective classes and return them
    Args:
        param1: str
    Returns:
        transformer classes 
    """
    tokenizer = BertTokenizer.from_pretrained(path_to_model)
    model = BertModel.from_pretrained(path_to_model, output_hidden_states=True)

    return tokenizer, model


def prepare_sentences(sentences, tokenizer, max_len=100):
    """Tokenize, pad, add special tokens
    Args:
        param1: list of strs
        param2: int
        param3: transformers tokenizer class object
    Returns:
        list of torch.Tensor
    """
    torch_sentences = []
    for sentence in sentences:
        torch_sentences.append(tokenizer.encode(sentence, add_special_tokens = True, max_length = max_len, pad_to_max_length = True, return_tensors = 'pt'))
    torch_sentences = torch.cat(torch_sentences, dim=0)

    return torch_sentences


def combine_srl_embs_bert_embs():
    """
    """
    comb_tensor = torch.cat((bert_tensor, srl_tensor), 0)
    pass


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fine_tune_BERT(model, tokenizer, config):
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
    criterion = nn.NLLLoss()
    print(6*"-" + "Checking which device to use..." + 6*"-")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
        print("device set to: CUDA -> using GPU #{}".format(gpu))
    else:
        device = torch.device("cpu")
        print("device set to: CPU")
    train_data, test_data = dataloader_XNLI(config[location]["path_XNLI"], tokenizer, config["batch_size"])
    optimizer = AdamW(model.parameters(),
            lr = 2e-5,
            eps = 1e-8
        )

    total_steps = len(train_data) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
            num_warmup_steps = 0,
            num_training_steps = total_steps
        )
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_data):
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            model.zero_grad()
            outputs = model(b_input_ids)
            import pdb; pdb.set_trace()
            loss = criterion(outputs, b_labels)
    
    


def main():
    args = parse_cmd_args()
    global location
    location = args.location
    config = load_json(args.config)
    tokenizer = BertTokenizer.from_pretrained(path_to_model)
    model = BertEntailmentClassifier(config[location]["path_BERT"])
    fine_tune_BERT(model, tokenizer, config)
    


if __name__ == "__main__":
    main()
