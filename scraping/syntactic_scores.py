import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel
from nltk.util import ngrams 
from collections import Counter
import regex as re 
import torch 
import torch.nn.functional as F
import spacy
from nltk.corpus import stopwords
from numpy import dot
from numpy.linalg import norm
import functorch
import numpy as np 
import random
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

all_content = pickle.load(open('all_data.pkl', 'rb'))
ngram_weights = [0.2, 0.8]

lang_nlp_map = {
    'english': 'en_core_web_sm',
    'german': 'de_core_news_sm',
    'portuguese': 'pt_core_news_sm',
}

lang_name_map = {
    'en': 'english',
    'de': 'german',
    'pt': 'portuguese',
}

class TfTokenizer:
    def __init__(self, lang):
        self.nlp = spacy.load(lang_nlp_map[lang])
        self.stopwords = stopwords.words(lang)
    def __call__(self, doc):
        doc = self.nlp(doc)
        return [token.lemma_ for token in doc if token.text not in self.stopwords]

def cosine_similarity(a, b):
    if dot(a, b)==0 or norm(a)==0 or norm(b)==0:
        return 0
    return dot(a, b)/(norm(a)*norm(b))

def compute_cosine_similarity(A, B):
    # Normalize rows of matrices A and B
    A_normalized = A / np.linalg.norm(A, axis=1)[:, np.newaxis]
    B_normalized = B / np.linalg.norm(B, axis=1)[:, np.newaxis]

    # Compute cosine similarity using dot product
    similarity_matrix = np.dot(A_normalized, B_normalized.T)

    return similarity_matrix

class TFScorer:
    def __init__(self, sents, obj_properties, sub_properties, language):
        self.sents = sents
        self.obj_properties = obj_properties
        self.sub_properties = sub_properties
        self.language = language
        self.ngram_weights = [0.2, 0.8]
        self.nlp = spacy.load(lang_nlp_map[self.language])
        self.vectorizer = CountVectorizer(tokenizer=TfTokenizer(self.language), ngram_range=(1, 1))

        self.processed_sents = [self._preprocess(sent) for sent in self.sents]
        self.processed_obj_properties = [self._preprocess(prop) for prop in self.obj_properties]
        self.processed_sub_properties = [self._preprocess(prop) for prop in self.sub_properties]    
        self._fit_vectorizer()

    def _preprocess(self, sent):
        return " ".join([token.lemma_.lower() for token in self.nlp(sent) if token.text not in stopwords.words(self.language)])
    
    def _fit_vectorizer(self):
        # all_sents = [sent for para in self.processed_sents for sent in para]
        # all_properties = [prop for para in self.processed_properties for prop in para]
        self.vectorizer.fit(self.sents + self.obj_properties + self.sub_properties)

    def _tf_scores(self):
        sent_vectors = self.vectorizer.transform(self.processed_sents).toarray()
        obj_prop_vectors = self.vectorizer.transform(self.processed_obj_properties).toarray()
        obj_similarity_matrix = compute_cosine_similarity(sent_vectors, obj_prop_vectors)
        sub_prop_vectors = self.vectorizer.transform(self.processed_sub_properties).toarray()
        sub_similarity_matrix = compute_cosine_similarity(sent_vectors, sub_prop_vectors)
            
        return obj_similarity_matrix, sub_similarity_matrix
    
    def compute_similarity(self):
        return self._tf_scores()

for lang in all_content:
    if(lang == 'en'):
        continue 
    keys = list(all_content[lang].keys())
    random.shuffle(keys)
    for title in keys[:50]:
        try:
            print(lang, title)
            if(lang == 'en'):
                sents = [sent for para in all_content[lang][title]['len_filtered_sentences'] for sent in para]
                sent_src = [i for i, para in enumerate(all_content[lang][title]['len_filtered_sentences']) for sent in para]
                obj_properties = [" ".join(fc) for fc in all_content[lang][title]['object_properties']]
                sub_properties = [" ".join(fc) for fc in all_content[lang][title]['subject_properties']]
                tf_scorer = TFScorer(sents, obj_properties, sub_properties, lang_name_map[lang])
                obj_similarity_matrix, sub_similarity_matrix = tf_scorer.compute_similarity()
                obj_para_scores = []
                sub_para_scores = []
                for i, obj_score, sub_score in zip(sent_src, obj_similarity_matrix, sub_similarity_matrix):
                    if(i >= len(obj_para_scores)):
                        obj_para_scores.append([])
                        sub_para_scores.append([])
                    obj_para_scores[i].append(obj_score)
                    sub_para_scores[i].append(sub_score)
                all_content[lang][title]['obj_syn_scores'] = obj_para_scores
                all_content[lang][title]['sub_syn_scores'] = sub_para_scores

            else:
                # sents = [sent for para in all_content[lang][title]['len_filtered_sentences'] for sent in para]
                # sent_src = [i for i, para in enumerate(all_content[lang][title]['len_filtered_sentences']) for sent in para]
                # trans_obj_properties = [" ".join(fc[1:]) for fc in all_content[lang][title]['translated_object_properties']]
                # trans_sub_properties = [" ".join(fc[:-1]) for fc in all_content[lang][title]['translated_subject_properties']]

                # tf_scorer = TFScorer(sents, trans_obj_properties, trans_sub_properties, lang_name_map[lang])
                # obj_similarity_matrix, sub_similarity_matrix = tf_scorer.compute_similarity()

                # obj_para_scores = []
                # sub_para_scores = []
                # for i, obj_score, sub_score in zip(sent_src, obj_similarity_matrix, sub_similarity_matrix):
                #     if(i >= len(obj_para_scores)):
                #         obj_para_scores.append([])
                #         sub_para_scores.append([])
                #     obj_para_scores[i].append(obj_score)
                #     sub_para_scores[i].append(sub_score)
                # all_content[lang][title]['obj_syn_scores'] = obj_para_scores
                # all_content[lang][title]['sub_syn_scores'] = sub_para_scores

                trans_sents = [sent for para in all_content[lang][title]['translated_sents'] for sent in para]
                sent_src = [i for i, para in enumerate(all_content[lang][title]['translated_sents']) for sent in para]
                obj_properties = [" ".join(fc) for fc in all_content[lang][title]['object_properties']]
                sub_properties = [" ".join(fc) for fc in all_content[lang][title]['subject_properties']]

                tf_scorer = TFScorer(trans_sents, obj_properties, sub_properties, 'english')
                obj_similarity_matrix, sub_similarity_matrix = tf_scorer.compute_similarity()
                obj_para_scores = []
                sub_para_scores = []
                for i, obj_score, sub_score in zip(sent_src, obj_similarity_matrix, sub_similarity_matrix):
                    if(i >= len(obj_para_scores)):
                        obj_para_scores.append([])
                        sub_para_scores.append([])
                    obj_para_scores[i].append(obj_score)
                    sub_para_scores[i].append(sub_score)
                all_content[lang][title]['obj_syn_scores'] = obj_para_scores
                all_content[lang][title]['sub_syn_scores'] = sub_para_scores
        except Exception:
            continue 

syn_scores_dump = open('non_eng_syn_scores_dump.pkl', 'wb')
pickle.dump(all_content, syn_scores_dump)
syn_scores_dump.close()
