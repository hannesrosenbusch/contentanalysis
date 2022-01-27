from sentence_transformers import SentenceTransformer
from contentanalysis import model_path
import fasttext.util
import numpy as np
import pickle
import re
import string

class quickie_classifier:
    
    def __init__(self, model_path = model_path):
        self.encoder = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
        fasttext.util.download_model('de', if_exists='ignore')
        self.encoder2 = fasttext.load_model("cc.de.300.bin") #heavy
        with open(model_path, "rb+") as f:
            self.model = pickle.load(f)
        self.labels = np.array(['arbeit','dating','eltern','essen','fashion','freunde','geld',
                                'gesundheit','internet','lesen','metaphysik','movie','musik',
                                'politik','reise','schlaf','schule','selfcare','shopping','spiele',
                                'sport','tiere','vehikel'])
    
    def preprocess_text(self, new_text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f" 
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        prepped_text = emoji_pattern.sub(r'', new_text)
        return prepped_text.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))
        
    def predict(self, new_text):
        prepped_text = self.preprocess_text(new_text)
        bert_features = self.encoder.encode(prepped_text)
        fasttext_features = np.array(self.encoder2.get_sentence_vector(prepped_text.replace("\n"," ")))
        features = np.concatenate((bert_features.reshape(1,-1), fasttext_features.reshape(1,-1)), axis = 1)
        y_pred = self.model.predict(features.reshape(1, -1))
        return self.labels[y_pred[0] == 1]
    
    def predict_proba(self, new_text):
        prepped_text = self.preprocess_text(new_text)
        bert_features = self.encoder.encode(prepped_text)
        fasttext_features = np.array(self.encoder2.get_sentence_vector(prepped_text.replace("\n"," ")))
        features = np.concatenate((bert_features.reshape(1,-1), fasttext_features.reshape(1,-1)), axis = 1)
        y_pred = self.model.predict_proba(features.reshape(1, -1))
        return y_pred
    
clf = quickie_classifier()      