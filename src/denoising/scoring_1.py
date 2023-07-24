import torch
import numpy as np
import tqdm 
import sys 
from transformers import AutoModelForCausalLM, AutoTokenizer,  AutoModelForSeq2SeqLM, T5EncoderModel

batch_size = 16
lang = sys.argv[1]
dataset = sys.argv[2]
model_name = sys.argv[3]
split_start = int(sys.argv[4])
split_end = int(sys.argv[5])

print(lang, dataset, model_name, split_start, split_end)

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

true_src = [f'generate {lang_map[lang]}: {i}' for i in open(f'/home2/aditya_hari/gsoc/data/{dataset}/{lang}/train_src', 'r').readlines()[split_start:split_end]]
true_tgt = open(f'/home2/aditya_hari/gsoc/data/{dataset}/{lang}/train_tgt', 'r').readlines()[split_start:split_end]

true_batches = [[true_src[i:i+batch_size] for i in range(0, len(true_src), batch_size)], [true_tgt[i:i+batch_size] for i in range(0, len(true_tgt), batch_size)]]
true_batches = list(zip(*true_batches))

true_scores = [] 
pb = tqdm.tqdm(total=len(true_batches))
for batch in true_batches:
    pb.update(1)
    with torch.no_grad():
        pro_score = to_tokens_and_logprobs(model, tokenizer, batch)
    true_scores.extend(pro_score)

with(open(f'scores/{dataset}/{model_name}_scores_{lang}_{str(split_start)}_{str(split_end)}', 'w')) as f:
    f.write('\n'.join([str(i) for i in true_scores]))
