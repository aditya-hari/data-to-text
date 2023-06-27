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
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

device = 'cuda'

all_content = pickle.load(open('all_data.pkl', 'rb'))

lang_map = {
    'english': 'en_core_web_sm',
    'german': 'de_core_news_sm',
    'portuguese': 'pt_core_news_sm',
}

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", padding='max_length', truncation='max_length', max_length=512)
model = AutoModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True).to('cuda')

def get_embedding(tokens, split_into_words=False):
    #print(tokens)
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=True, truncation=True, max_length=512, is_split_into_words=split_into_words, return_tensors="pt").to(device)
        #print(tokenizer.convert_ids_to_tokens(tokenized_facts['input_ids'][0]))
        states = model(**tokenized_facts).hidden_states
        output = torch.stack([states[i] for i in range(len(states))])
        output = output.squeeze()
        #print(output.shape)
        final_hidden_state = torch.mean(output[:, :, ...], dim=0)
        #final_hidden_state = output[-2, :, ...]
        #print(final_hidden_state.shape)
        return final_hidden_state[1:-1]
        #return embeddings['last_hidden_state'] #tokenized_facts['attention_mask']
        #return torch.mean(embeddings['last_hidden_state'], dim=1)

vector_similarity = lambda factemb, sentemb: functorch.vmap(lambda row_a: F.cosine_similarity(row_a, factemb))(sentemb).T

for lang in all_content:
    print(lang)
    for title in all_content[lang]:
        print(title)
        if(lang == 'en'):
            semantic_scores = {'obj': [], 'subj': []}
            for para in all_content[lang][title]['len_filtered_sentences'][:5]:
                para_scores_obj = []
                para_scores_subj = []
                for sent in para:
                    sent_scores = []
                    for obj_prop in all_content[lang][title]['object_properties']:
                        fact_str = " ".join((title, obj_prop[0], obj_prop[1]))
                        sent_scores.append(vector_similarity(get_embedding(fact_str), get_embedding(sent)).cpu())
                    para_scores_obj.append(sent_scores)
                semantic_scores['obj'].append(para_scores_obj)

                for sent in para:
                    sent_scores = []
                    for subj_prop in all_content[lang][title]['subject_properties']:
                        fact_str = " ".join((subj_prop[1], subj_prop[0], title))
                        sent_scores.append(vector_similarity(get_embedding(fact_str), get_embedding(sent)).cpu())
                    para_scores_subj.append(sent_scores)
                semantic_scores['subj'].append(para_scores_subj)
            all_content[lang][title]['sent_sem_scores'] = semantic_scores

        else:
            semantic_scores = {'obj': [], 'subj': []}
            for para in all_content[lang][title]['len_filtered_sentences'][:5]:
                para_scores_obj = []
                para_scores_subj = []
                for sent in para:
                    sent_scores = []
                    for obj_prop in all_content[lang][title]['object_properties']:
                        fact_str = " ".join((title, obj_prop[0], obj_prop[1]))
                        sent_scores.append(vector_similarity(get_embedding(obj_prop), get_embedding(sent)).cpu())
                    para_scores_obj.append(sent_scores)
                semantic_scores['obj'].append(para_scores_obj)

                for sent in para:
                    sent_scores = []
                    for subj_prop in all_content[lang][title]['subject_properties']:
                        fact_str = " ".join((subj_prop[1], subj_prop[0], title))
                        sent_scores.append(vector_similarity(get_embedding(subj_prop), get_embedding(sent)).cpu())
                    para_scores_subj.append(sent_scores)
                semantic_scores['subj'].append(para_scores_subj)
            all_content[lang][title]['sent_sem_scores'] = semantic_scores

            # trans_semantic_scores = {'obj': [], 'subj': []}
            # for para in all_content[lang][title]['translated_sents'][:5]:
            #     para_score_obj = []
            #     para_score_subj = []
            #     for sent in para:
            #         sent_scores = []
            #         for obj_prop in all_content[lang][title]['translated_object_properties']:
            #             fact_str = " ".join(obj_prop)
            #             sent_scores.append(vector_similarity(get_embedding(fact_str), get_embedding(sent)).cpu())
            #         para_score_obj.append(sent_scores)
            #     trans_semantic_scores['obj'].append(para_score_obj)

            #     for sent in para:
            #         sent_scores = []
            #         for subj_prop in all_content[lang][title]['translated_subject_properties']:
            #             fact_str = " ".join(subj_prop)
            #             sent_scores.append(vector_similarity(get_embedding(fact_str), get_embedding(sent)).cpu())
            #         para_score_subj.append(sent_scores)
            #     trans_semantic_scores['subj'].append(para_score_subj)
            # all_content[lang][title]['trans_sent_sem_scores'] = trans_semantic_scores
            
semantic_dump = open('all_data_semantic_scores.pkl', 'wb')
pickle.dump(all_content, semantic_dump)
semantic_dump.close()