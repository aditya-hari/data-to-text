from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
import tqdm 

nllb_tokenizer_ga = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='gle_Latn')
nllb_tokenizer_de = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='deu_Latn')
nllb_tokenizer_en = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='eng_Latn')

nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to('cuda')
nllb_model.eval()

lang_map = {
    'br': 'bre_Latn',
    'cy': 'cym_Latn',
    'ga': 'gle_Latn',
    'mt': 'mlt_Latn',
    'ru': 'rus_Cyrl'
}

candidates = json.load(open('/scratch/candidates.json', 'r'))

all_sents = {'de': [], 'ga': []}
all_sents_src = {'de': [], 'ga': []}

all_props = {'de': [], 'ga': []}
all_props_src = {'de': [], 'ga': []}

word_length = lambda x: len(x.split())  

for key, value in candidates.items():
  props = [' | '.join(prop) for prop in value['properties']]

  if('de_text' in value):
    texts_retained = [txt for txt in value['de_text'] if word_length(txt) > 10 and word_length(txt) < 256]
    all_sents['de'].extend(texts_retained)
    all_sents_src['de'].extend([key for _ in range(len(texts_retained))])

    all_props['de'].extend(props)
    all_props_src['de'].extend([key for _ in range(len(props))])

  if('ga_text' in value):
    texts_retained = [txt for txt in value['ga_text'] if word_length(txt) > 10 and word_length(txt) < 256]
    all_sents['ga'].extend(texts_retained)
    all_sents_src['ga'].extend([key for _ in range(len(texts_retained))])

    all_props['ga'].extend(props)
    all_props_src['ga'].extend([key for _ in range(len(props))])

sents_out = {'de': [], 'ga': []}
props_out = {'de': [], 'ga': []}

for lang in all_sents:
  if(lang == 'de'):
    tok = nllb_tokenizer_de
  else:
    tok = nllb_tokenizer_ga
  out_file = open(f'./translated_sents_{lang}.txt', 'w')

  sents_batched = [all_sents[lang][i:i+32] for i in range(0, len(all_sents[lang]), 32)]
  pb = tqdm.tqdm(range(len(sents_batched)))
  for i, batch in enumerate(sents_batched):
    pb.update(1)
    inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
    translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=tok.lang_code_to_id['eng_Latn'], max_length=256)
    out = tok.batch_decode(translated_tokens, skip_special_tokens=True)
    out_file.write('\n'.join(out) + '\n')
    sents_out[lang].extend(out)
  out_file.close()
