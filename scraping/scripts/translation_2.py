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

candidates = json.load(open('/home2/aditya_hari/gsoc/rdf-to-text/scraping/notebooks/filtered_candidates.json', 'r'))

all_sents = {'de': [], 'ga': []}
all_sents_src = {'de': [], 'ga': []}

all_props = {'de': [], 'ga': []}
all_props_src = {'de': [], 'ga': []}

word_length = lambda x: len(x.split())  

for key, value in list(candidates.items())[200000:300000]:
  props = [' | '.join(prop) for prop in value['properties']]

  if('de_text' in value):
    all_sents['de'].extend(value['de_text'])
    all_props['de'].extend(props)

  if('ga_text' in value):
    all_sents['ga'].extend(value['ga_text'])
    all_props['ga'].extend(props)

sents_out = {'de': [], 'ga': []}
props_out = {'de': [], 'ga': []}

for lang in all_sents:
  if(lang == 'de'):
    tok = nllb_tokenizer_de
  else:
    tok = nllb_tokenizer_ga
  out_file = open(f'/scratch/sents/2_translated_sents_{lang}.txt', 'w')

  sents_batched = [all_sents[lang][i:i+32] for i in range(0, len(all_sents[lang]), 32)]
  pb = tqdm.tqdm(range(len(sents_batched)))
  for i, batch in enumerate(sents_batched):
    pb.update(1)
    inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
    translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=tok.lang_code_to_id['eng_Latn'], max_length=256)
    out = tok.batch_decode(translated_tokens, skip_special_tokens=True)
    pairs = list(zip(batch, out))
    for pair in pairs:
      out_file.write(f'{pair[0]} @@@ {pair[1]}\n')
    sents_out[lang].extend(out)
  out_file.close()
