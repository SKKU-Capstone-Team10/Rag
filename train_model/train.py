from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModel, AutoTokenizer
import torch
import random
import os

dataset = load_dataset("virattt/financial-qa-10K", split="train")
qa_pairs = [(item['question'], item['answer']) for item in dataset]

triplets = []
num_examples = len(qa_pairs)
for idx, (q, pos) in enumerate(qa_pairs):
    neg_idx = random.choice([i for i in range(num_examples) if i != idx])
    neg = qa_pairs[neg_idx][1]
    triplets.append(InputExample(texts=[q, pos, neg]))

model_name = "BAAI/bge-large-en"
base_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)
model_with_lora = get_peft_model(base_model, peft_config)

bi_encoder = SentenceTransformer(model_name)
bi_encoder._first_module().auto_model = model_with_lora

train_dataloader = DataLoader(triplets, shuffle=True, batch_size=32)
train_loss = losses.TripletLoss(model=bi_encoder)

bi_encoder.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path="./../model/bge_large_lora_full"  
)

base_model.save_pretrained("./../model/bge_large_base")
tokenizer.save_pretrained("./../model/bge_large_base")
model_with_lora.save_pretrained("./../model/bge_large_lora_adapter")