import torch
import transformers
from transformers import BertTokenizer, BertModel, BertConfig , BertForSequenceClassification
import numpy as np



class EmoClassifier:


    def __init__(self) -> None:

        self.tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert_base_uncased')
        self.model.eval()
        self.labels = ['anger','confusion', 'curiosity', 'desire', 'digust', 'embarrassment', 'fear', 'joy', 'love', 'neutral', 'optimism', 'pride', 'sadness', 'surprise']


    def predict_emo(self,text):

        input = self.tokenizer.encode_plus(text,
                                      None,
                                      add_special_tokens=True,
                                      max_length=128,
                                      pad_to_max_length=True,
                                      return_token_type_ids=True,
                                      return_tensors='pt')

        logits = self.model(**input)['logits']
        proba = torch.nn.functional.softmax(logits, dim=1).detach().numpy()
        idx = np.argmax(proba) 
        pred = self.labels[idx]

        return (text, pred, np.max(proba))




