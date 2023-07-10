from sklearn.feature_extraction.text import CountVectorizer
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
import tqdm 

nlp = spacy.load('xx_sent_ud_sm')
nlp.add_pipe('sentencizer')

label_map = {} 
with open('/scratch/useful/subject_set_labels.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        label_map.update(line)

prop_dict = {} 
with open('/scratch/useful/ontology_props.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        prop_dict.update({line['name']: line['properties']})

type_dict = {} 
with open('/scratch/useful/mapping_transitive.ttl', 'r') as f:
    for line in f:
        line = line.split()
        if(line[0] not in type_dict):
            type_dict[line[0]] = set() 
        type_dict[line[0]].add(line[2].split('/')[-1])

def get_prop_name(item):
    if("#literal" in item):
        return item.split('#literal')[0], True
    if(item in label_map):
        tail_name = label_map[item]
    else:
        tail_name = re.sub('_', ' ',item.split('/')[-1])
    return tail_name, False

wanted_types = ["Place", "Person", "Organization", "Organisation"]

langs = ['ga', 'de', 'en']
candidates = {} 

for lang in langs:
    print(lang)
    with(open(f'/scratch/useful/abstracts_{lang}.jsonl', 'r')) as f:
        pb = tqdm.tqdm(total=1000000)
        for i, line in enumerate(f):
            pb.update(1)
            item = json.loads(line)
            rsc = item['resource']
            txt = item['text']
            found = False 

            for t in wanted_types:
                if(f'{t}>' in type_dict[f'<{rsc}>']):
                    found = True 
            
            if(not(found)):
                continue  

            name = get_prop_name(rsc)[0]
            if(name not in candidates):
                candidates[name] = {}

                properties = [] 
                fw_props = prop_dict[rsc]['properties']
                for prop in fw_props:
                    for item in fw_props[prop]:
                        item_name = get_prop_name(item)[0]
                        properties.append((name, prop, item_name))
                
                rv_props = prop_dict[rsc]['reverse_properties']
                for prop in rv_props:
                    for item in rv_props[prop][:3]:
                        item_name = get_prop_name(item)[0]
                        properties.append((item_name, prop, name))
                
                candidates[name]['properties'] = properties

            doc = nlp(txt)
            sents = [] 
            filtered_sents = []
            for sent in doc.sents:
                sents.append(sent.text)
            
            if(lang!='en'):
                merged_sents = [sents[0]]
                for sent in sents[1:]:
                    if(re.search(r'\d+\.$', sent)):
                        merged_sents[-1] += sent
                    else:
                        merged_sents.append(sent)
            else:
                merged_sents = sents
            
            for sent in merged_sents:
                if(len(sent.split()) > 5 and len(sent.split()) < 250):
                    filtered_sents.append(sent)
            candidates[name][f'{lang}_text'] = filtered_sents

save_name = 'filtered_candidates.json'
with open(save_name, 'w') as f:
    json.dump(candidates, f)