from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."


def bert_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    return output


from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
text = "Replace me by any text you'd like."


def Roberta_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    return output


from transformers import BartTokenizer, BartModel

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartModel.from_pretrained("facebook/bart-base")
text = "Replace me by any text you'd like."


def bart_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    return output
