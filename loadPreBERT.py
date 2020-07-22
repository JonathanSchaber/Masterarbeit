from transformers import BertTokenizer, BertForMaskedLM


def create_model_and_tokenizer():
    """Create respective classes and return them
    Args:
        param1: str
    Returns:
        transformer classes 
    """
    tokenizer = BertTokenizer.from_pretrained(path_to_model)
    model = BertForMaskedLM.from_pretrained(path_to_model)

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
        torch_sentences.append(tokenizer.encode(sentence, add_special_tokens = True, max_length = max_len, pad_to_max_length = True, return_tensors = 'pt')

    return torch_sentences


