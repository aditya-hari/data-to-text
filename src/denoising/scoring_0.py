import torch
import numpy as np
import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer,  AutoModelForSeq2SeqLM, T5EncoderModel

batch_size = 24
lang = 'ga'
dataset = 'processed'
model_name = 'syn'

lang_map = {
    'en': 'English',
    'ga': 'Irish',
    'de': 'German'
}

tokenizer = AutoTokenizer.from_pretrained('google/mt5-small', padding=True, truncation=True, model_max_length=512)
tokenizer.add_special_tokens({'additional_special_tokens': ['<TSP>']})

model = AutoModelForSeq2SeqLM.from_pretrained(f'/scratch/mt5/mt5_{model_name}', load_in_4bit=True)
model.resize_token_embeddings(len(tokenizer))
model.eval()

def to_tokens_and_logprobs(model, tokenizer, input_texts, device='cuda'):
    input_ids_ = tokenizer(input_texts[0], text_target=input_texts[1], padding=True, truncation=True, return_tensors="pt").to(device)
    label_ids = input_ids_.labels
    input_ids = input_ids_.input_ids
    outputs = model(**input_ids_)

    probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    probs = probs[:, :, :]
    input_ids = label_ids[:, :]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    return torch.mean(gen_probs, dim=1).detach().cpu().numpy()

true_src = [f'generate {lang_map[lang]}: {i}' for i in open(f'/home2/aditya_hari/gsoc/data/{dataset}/{lang}/train_src', 'r').readlines()]
true_tgt = open(f'/home2/aditya_hari/gsoc/data/{dataset}/{lang}/train_tgt', 'r').readlines()

true_batches = [[true_src[i:i+batch_size] for i in range(0, len(true_src), batch_size)], [true_tgt[i:i+batch_size] for i in range(0, len(true_tgt), batch_size)]]
true_batches = list(zip(*true_batches))

true_scores = [] 
pb = tqdm.tqdm(total=len(true_batches))
for batch in true_batches:
    pb.update(1)
    with torch.no_grad():
        pro_score = to_tokens_and_logprobs(model, tokenizer, batch)
    true_scores.extend(pro_score)

with(open(f'scores/{dataset}/{model_name}_scores_{lang}', 'w')) as f:
    f.write('\n'.join([str(i) for i in true_scores]))
