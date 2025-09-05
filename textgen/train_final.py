import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import os
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Load the Base Tokenizer and Model 
model_name = 'gpt2'
print(f"Loading base tokenizer and model for: {model_name}")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Define and Add Special Tokens
special_tokens_dict = {
    'eos_token': '<|endofstory|>',
    'pad_token': tokenizer.eos_token
}
print("Adding new special tokens to the tokenizer...")
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
print(f"{num_added_toks} new tokens added. New vocabulary size: {len(tokenizer)}")

# 3. Load and Prepare the Dataset 
dataset_path = 'stories.txt'
if not os.path.exists(dataset_path):
    colab_path = f"/content/drive/My Drive/textgen/{dataset_path}"
    if os.path.exists(colab_path):
        dataset_path = colab_path
    else:
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path} or {colab_path}")

print(f"Loading and preparing dataset from: {dataset_path}")

with open(dataset_path, 'r', encoding='utf-8') as f:
    text = f.read()

stories = text.split('\n\n')
stories = [s.strip() for s in stories if s.strip()]
formatted_stories = [f"{story} {tokenizer.eos_token}" for story in stories]

subset_stories = formatted_stories[:2000]
dataset = Dataset.from_dict({'text': subset_stories})

print(f"Loaded {len(subset_stories)} stories for training (subset). Formatting and tokenizing...")
tokenized_datasets = dataset.map(
    lambda examples: tokenizer(examples['text'], truncation=True, max_length=256),  # reduced length
    batched=True,
    remove_columns=['text']
)

# 4. Define Training Arguments and Trainer 
training_args = TrainingArguments(
    output_dir='./results_final',
    num_train_epochs=5,   
    per_device_train_batch_size=8,
    warmup_steps=100,    
    weight_decay=0.01,
    logging_dir='./logs_final',
    logging_steps=20,     
    save_steps=500,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# 5. Start the Training 
print("\nStarting AI Storyteller training...")
trainer.train()

# 6. Save the Final Model 
output_dir = './results_final/final_model'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nAI Storyteller training complete!")
print(f"Model and tokenizer saved in '{output_dir}' folder.")
