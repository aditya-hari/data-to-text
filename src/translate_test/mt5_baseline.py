from glob import glob 
from transformers import MT5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from datasets import load_dataset
import regex as re
from sklearn.model_selection import train_test_split

lang_codes = {
    "cy": "Welsh",
    "br": "Breton",
    "ga": "Irish",
    "mt": "Maltese",
    "ru": "Russian",
}

config_dict = {"TRAIN_BATCH_SIZE": 1, 
            "VALID_BATCH_SIZE": 1, 
            "LEARNING_RATE":1e-5, 
            "CLASS_WEIGHTS": 0, 
            "EPOCHS": 10, 
            "WT_DECAY":0}

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

lang_files = glob("./data/processed/*")

train_srcs = [] 
train_tgts = []

eval_srcs = [] 
eval_tgts = []

test_srcs = []
test_tgts = []

dataset = load_dataset("web_nlg", "release_v3.0_ru")

for ex in dataset["train"]:
    triple = ' <TSP> '.join(ex['modified_triple_sets']['mtriple_set'][0])
    for lang, sents in zip(ex['lex']['lang'], ex['lex']['text']):
        if(lang == 'ru'):
            train_srcs.append(triple)
            train_tgts.append(re.sub(r"[ ]{2,}", " " , sents).strip())

for ex in dataset["dev"]:
    triple = ' <TSP> '.join(ex['modified_triple_sets']['mtriple_set'][0])
    for lang, sents in zip(ex['lex']['lang'], ex['lex']['text']):
        if(lang == 'ru'):
            eval_srcs.append(triple)
            eval_tgts.append(re.sub(r"[ ]{2,}", " " , sents).strip())

for ex in dataset["test"]:
    triple = ' <TSP> '.join(ex['modified_triple_sets']['mtriple_set'][0])
    for lang, sents in zip(ex['lex']['lang'], ex['lex']['text']):
        if(lang == 'ru'):
            test_srcs.append(triple)
            test_tgts.append(re.sub(r"[ ]{2,}", " " , sents).strip())


print(train_srcs[:5])
print(train_tgts[:5])

# for lang_file in lang_files:
#     lang = lang_file.split("/")[-1]
#     if(lang != 'ga'):
#         continue
#     train_src = open(f'{lang_file}/train_src', 'r').readlines()
#     train_tgt = open(f'{lang_file}/train_tgt', 'r').readlines()
#     train_srcs.extend([re.sub(r"[ ]{2,}", " " , line).strip() for line in train_src])
#     train_tgts.extend([line.strip() for line in train_tgt])

#     eval_src = open(f'{lang_file}/eval_src', 'r').readlines()
#     eval_tgt = open(f'{lang_file}/eval_tgt', 'r').readlines()
#     eval_srcs.extend([re.sub(r"[ ]{2,}", " " , line).strip() for line in eval_src])
#     eval_tgts.extend([line.strip() for line in eval_tgt])

# eval_srcs, test_srcs, eval_tgts, test_tgts = train_test_split(eval_srcs, eval_tgts, test_size=0.4, random_state=42)

class Text2TextDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]

        input_encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=400,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer.encode_plus(
            target_text,
            max_length=400,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


train_dataset = Text2TextDataset(train_srcs, train_tgts, tokenizer)
eval_dataset = Text2TextDataset(eval_srcs, eval_tgts, tokenizer)
test_dataset = Text2TextDataset(test_srcs, test_tgts, tokenizer)

training_args = TrainingArguments(
    output_dir='/scratch/aditya_hari/gsoc/mt5-ru',  
    num_train_epochs=config_dict['EPOCHS'],              
    per_device_train_batch_size=config_dict["TRAIN_BATCH_SIZE"],  
    per_device_eval_batch_size=config_dict["VALID_BATCH_SIZE"],   
    warmup_steps=0,                
    weight_decay=config_dict["WT_DECAY"],              
    logging_dir='/scratch/aditya_hari/gsoc/mt5-ru',            
    logging_steps=250,
    save_strategy='epoch',
    save_total_limit=3,
    evaluation_strategy="epoch", 
    learning_rate = config_dict["LEARNING_RATE"],
    metric_for_best_model = 'eval_loss',
    load_best_model_at_end = True,
    report_to='wandb',
    #lr_scheduler_type="constant",
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()