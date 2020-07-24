from transformers import (
        BertTokenizer,
        BertModel,
        BertForQuestionAnswering
        )



def create_model_and_tokenizer(path_to_model):
    """Create respective classes and return them
    Args:
        param1: str
    Returns:
        transformer classes 
    """
    tokenizer = BertTokenizer.from_pretrained(path_to_model)
    model = BertModel.from_pretrained(path_to_model)

    return tokenizer, model


def prepare_sentences(sentences, max_len):
    """Tokenize, pad, add special tokens
    Args:
        param1: list of strs
        param2: int
    Returns:
        list of torch.Tensor
    """
    torch_sentences = []
    for sentence in sentences:
        torch_sentences.append(tokenizer.encode(sentence, add_special_tokens = True, max_length = max_len, pad_to_max_length = True, return_tensors = 'pt'))
    torch_sentences = torch.cat(torch_sentences, dim=0)

    return torch_sentences

def main():
    # path = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/germanBERT/bert-base-german-cased-pytorch_model.bin"
    path = "/home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/germanBERT/"
    tokenizer, model = create_model_and_tokenizer(path)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
