import torch

from torch import nn
from transformers import (
        BertTokenizer,
        BertModel,
        BertForQuestionAnswering
        )

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


def main():
    path = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/germanBERT/"
    tokenizer, model = create_model_and_tokenizer(path)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
