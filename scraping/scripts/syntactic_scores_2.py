from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.util import ngrams 
from nltk.tokenize import sent_tokenize
from collections import Counter
import regex as re 
from numpy import dot
from numpy.linalg import norm
import numpy as np 
import json 
import random 
import spacy 
import glob 
import tqdm 

candidates = json.load(open('/home2/aditya_hari/gsoc/rdf-to-text/scraping/notebooks/filtered_candidates.json', 'r'))
# regex pattern to split at camel case
pattern = re.compile(r'(?<!^)(?=[A-Z])')

# Function to compute average TF-IDF similarity between every pair of sentences in two lists of sentences
def get_similarity(sent_list1, sent_list2):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    sent_list1 = [' '.join([i for i in pattern.split(sent)]) for sent in sent_list1]
    sent_list2 = [' '.join([i for i in pattern.split(sent)]) for sent in sent_list2]
    X1 = vectorizer.fit_transform(sent_list1)
    X2 = vectorizer.transform(sent_list2)
    sim_mat = np.zeros((len(sent_list1), len(sent_list2)))
    for i in range(len(sent_list1)):
        for j in range(len(sent_list2)):
            if(norm(X1[i].toarray()[0])*norm(X2[j].toarray()[0]) == 0):
                sim_mat[i][j] = 0
            else:
                sim_mat[i][j] = dot(X1[i].toarray()[0], X2[j].toarray()[0])/(norm(X1[i].toarray()[0])*norm(X2[j].toarray()[0]))
    return sim_mat 

filtered_candidates = {} 
keys = sorted(list(candidates.keys()))

pb = tqdm.tqdm(total=len(keys[200000:300000]))
for key in keys[200000:300000]:
    pb.update(1)
    try:
        all_prop_strs = [] 
        props_filtered = [prop for prop in candidates[key]['properties'] if('' not in prop)]
        for prop in props_filtered:
            prop_str = ' '.join([' '.join(pattern.split(re.sub(r'[^\w]+', ' ', i))) for i in prop])
            all_prop_strs.append(prop_str)
        if('en_text' in candidates[key] and len(candidates[key]['en_text']) != 0):
            en_similarity_mat = get_similarity(all_prop_strs, candidates[key]['en_text'])
            en_above_thresh = np.where(en_similarity_mat > 0)
            if(len(en_above_thresh[1]) != 0):
                # en_retained_props = [[] for _ in range(len(candidates[key]['en_text']))]
                # for sent_idx, prop_idx in zip(en_above_thresh[1], en_above_thresh[0]):
                #     en_retained_props[sent_idx].append(props_filtered[prop_idx])
                if(key not in filtered_candidates):
                    filtered_candidates[key] = candidates[key]
                # filtered_candidates[key]['en_text'] = candidates[key]['en_text']
                # filtered_candidates[key]['en_retained_props'] = en_retained_props
                filtered_candidates[key]['en_sim_mat'] = en_similarity_mat.tolist()

        if('de_translated' in candidates[key] and len(candidates[key]['de_translated']) != 0):
            de_similarity_mat = get_similarity(all_prop_strs, candidates[key]['de_translated'])
            de_above_thresh = np.where(de_similarity_mat > 0)
            if(len(de_above_thresh[1]) != 0):
                # de_retained_props = [[] for _ in range(len(candidates[key]['de_translated']))]
                # for sent_idx, prop_idx in zip(de_above_thresh[1], de_above_thresh[0]):
                #     de_retained_props[sent_idx].append(props_filtered[prop_idx])
                if(key not in filtered_candidates):
                    filtered_candidates[key] = candidates[key]
                # filtered_candidates[key]['de_translated'] = candidates[key]['de_translated']
                # filtered_candidates[key]['de_retained_props'] = de_retained_props
                filtered_candidates[key]['de_sim_mat'] = de_similarity_mat.tolist()
        
        if('ga_translated' in candidates[key] and len(candidates[key]['ga_translated']) != 0):
            ga_similarity_mat = get_similarity(all_prop_strs, candidates[key]['ga_translated'])
            ga_above_thresh = np.where(ga_similarity_mat > 0)
            if(len(ga_above_thresh[1]) != 0):
                # ga_retained_props = [[] for _ in range(len(candidates[key]['ga_translated']))]
                # for sent_idx, prop_idx in zip(ga_above_thresh[1], ga_above_thresh[0]):
                #     ga_retained_props[sent_idx].append(props_filtered[prop_idx])
                if(key not in filtered_candidates):
                    filtered_candidates[key] = candidates[key]
                # filtered_candidates[key]['ga_translated'] = candidates[key]['ga_translated']
                # filtered_candidates[key]['ga_retained_props'] = ga_retained_props
                filtered_candidates[key]['ga_sim_mat'] = ga_similarity_mat.tolist()
    except Exception as e:
        print(e)
        continue
    
with(open('props/prop_candidates_2.json', 'w')) as f:
    json.dump(filtered_candidates, f)
