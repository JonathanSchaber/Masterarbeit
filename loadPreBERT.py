from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained(path_to_model)

model = BertForMaskedLM.from_pretrained(path_to_model)

