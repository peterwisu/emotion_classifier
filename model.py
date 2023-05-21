import torch
import transformers
from transformers import BertTokenizer, BertModel, BertConfig , BertForSequenceClassification , RobertaForSequenceClassification, RobertaTokenizerFast
import numpy as np
import pickle
import sklearn

class EmoClassifier:


    def __init__(self,logger, model_type='logistic') -> None:
        
        
        self.model_type = model_type 
        if self.model_type == "roberta":
            print("Loading Roberta tokenizer.....")
            #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer  = RobertaTokenizerFast.from_pretrained('roberta-base')
            print("Loading Roberta Model......")
            #self.model =  BertForSequenceClassification.from_pretrained('bert-base-uncased')
            
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=14)
            self.model.load_state_dict(torch.load("./ckpt/best_model.pt", map_location=torch.device('cpu')))
            
            
            self.model.eval()
            
            self.labels = ['anger', 'confusion', 'curiosity', 'desire', 'disgust', 'embarrassment', 'fear', 'joy', 'love', 'optimism', 'pride', 'sadness', 'surprise', 'neutral']
        else:
            
            print("Loading Logistic Regression Tokenizer........")
            
            with open('./ckpt/tokenizer_bow.pkl', 'rb') as file:
                
                self.tokenizer = pickle.load(file)
            
            print("Loading Logistic Model.........")
            
            with open('./ckpt/model_logreg_sgd.pkl', 'rb') as file:
                
                self.model = pickle.load(file)
                
                
            self.labels = ['anger','confusion', 'curiosity', 'desire', 'digust', 'embarrassment', 'fear', 'joy', 'love', 'neutral', 'optimism', 'pride', 'sadness', 'surprise']
            
        self.logger = logger

    def predict_emo(self,text):

        if self.model_type =="roberta":
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
        
        else :
            
            input = self.tokenizer.transform([text])
            proba = self.model.predict_proba(input)
            idx = np.argmax(proba)
            pred = self.labels[idx]
        
        self.logger.info("Logging additional Information", extra={'user_input': text,
                                                                  'model_prediction': pred,
                                                                  'score': np.max(proba)})


        return (text, pred, np.max(proba))




