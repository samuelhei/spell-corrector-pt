import unicodedata
import re
import os
import itertools
import numpy as np
import scipy
from joblib import dump, load

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PathNotExistsException(Exception):
    pass

class NotTrainedModelException(Exception):
    pass

class SpellCorrector:

    def __init__(self, dictionary=[], ngram_range=(1,3)):
        self.dictionary = dictionary
        self.ngram_range = ngram_range
        self.clean_dictionary_method()

    def clean_dictionary_method(self):
        self.clean_dictionary = list(map(self.clean_text, self.dictionary))    

    def clean_text(self, text):
        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn')

        text = re.sub('[^a-z ]', '', text)
        return text

    def invert_text(self,text):
        text_l = list(text)
        text_l.reverse()
        return ''.join(text_l)

    def sub_phonetic(self,text):
        text = ''.join(ch for ch, _ in itertools.groupby(text))

        subs = [
            (['BL','BR'],'B'),
            (['PH'],'F'),
            (['GL', 'GR', 'MG', 'NG', 'RG'],'G'),
            (['Y'],'I'),
            (['GE', 'GI', 'RJ', 'MJ'],'J'),
            (['CA', 'CO', 'CU', 'CK', 'Q'],'K'),
            (['N'],'M'),
            (['AO', 'AUM', 'GM', 'MD', 'OM', 'ON'],'M'),
            (['PR'],'P'),
            (['L'],'R'),
            (['CE', 'CI', 'CH', 'CS', 'RS', 'TS', 'X', 'Z', 'C'],'S'),
            (['TR', 'TL', 'CT', 'RT', 'ST', 'PT'],'T')
        ]
        
        for sub_list, sub in subs:
            for check in sub_list:
                text = text.replace(check.lower(), sub.lower())      
        return text

    def train(self):
        self.phonetic = []
        self.comparation_dictionary = []

        for word in self.clean_dictionary:
            self.comparation_dictionary.append(word + self.invert_text(word))
            self.phonetic.append(self.sub_phonetic(word))
        
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=self.ngram_range)
        self.similarity_vector = self.vectorizer.fit_transform(self.comparation_dictionary)

    def get_correct_word(self, word, return_probability=False):
        def result(word, probability):
            if return_probability:
                return word, probability
            else:
                return word
        
        self.check_model()

        clean_word = self.clean_text(word)
        phonetic = self.sub_phonetic(clean_word)
        test_word = word + self.invert_text(word)
        
        if clean_word in self.clean_dictionary:
            return result(self.dictionary[self.clean_dictionary.index(clean_word)] , 1)
 
        if phonetic in self.phonetic:
            return result(self.dictionary[self.phonetic.index(phonetic)], 0.9)
    
        X = self.vectorizer.transform([test_word])
        similarity = cosine_similarity(X, self.similarity_vector)
        key = np.argmax(similarity)
        return result(self.dictionary[key], np.max(similarity))

    def check_model(self):
        tests = ['phonetic','comparation_dictionary','dictionary']
        
        for t in tests:
            if len(getattr(self, t)) <= 0:
                raise NotTrainedModelException(
                    'Model not trained "{}" is not valid (maybe you need to train the model again)'.format(t))
    
        for t in ['vectorizer', 'similarity_vector']:
            if t not in dir(self):
                raise NotTrainedModelException(
                    'Model not trained "{}" is not valid (maybe you need to train the model again)'.format(t))


    def dump_model(self, full_path):
        if os.path.isdir(full_path) == False:
            raise PathNotExistsException('"{}" path do not exists'.format(full_path))
        
        self.check_model()

        with open(os.path.join(full_path, 'phonetic.txt'), 'w') as f:
            f.write("\n".join(self.phonetic))

        with open(os.path.join(full_path, 'comparation_dictionary.txt'), 'w') as f:
            f.write("\n".join(self.comparation_dictionary))
    
        with open(os.path.join(full_path, 'dictionary.txt'), 'w') as f:
            f.write("\n".join(self.dictionary))
    
        dump(self.vectorizer, os.path.join(full_path, 'vectorizer.joblib')) 
        scipy.sparse.save_npz(os.path.join(full_path, 'similarity_vector.npz'), self.similarity_vector)

    def load_model(self, full_path):
        if os.path.isdir(full_path) == False:
            raise PathNotExistsException('"{}" path do not exists'.format(full_path))

        with open(os.path.join(full_path, 'phonetic.txt'), 'r') as f:
            self.phonetic = f.read().split("\n")

        with open(os.path.join(full_path, 'comparation_dictionary.txt'), 'r') as f:
            self.comparation_dictionary = f.read().split("\n")
    
        with open(os.path.join(full_path, 'dictionary.txt'), 'r') as f:
            self.dictionary = f.read().split("\n")
    
        self.vectorizer = load(os.path.join(full_path, 'vectorizer.joblib')) 
        self.similarity_vector = scipy.sparse.load_npz(os.path.join(full_path, 'similarity_vector.npz'))
        self.check_model()
    