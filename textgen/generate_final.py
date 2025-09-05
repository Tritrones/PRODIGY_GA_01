import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
import os

logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Load the Fine-tuned Model and Tokenizer 

model_path = "./results_final/final_model"

# Check if the model directory exists
if not os.path.exists(model_path):
    print("Error: Model directory not found. Please run train_final.py first to create the model at: results_storyteller/final_model")
    exit()

print(f"Loading AI Storyteller from: {model_path}")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 2. Interactive Loop 
print("\n🤖 AI Storyteller is ready! Type a prompt to start a story.")
print("   Type 'quit' or 'exit' to end the session.")

while True:
    prompt = input("\n>> Your prompt: ")

    if prompt.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    #  3. Encode the Prompt 
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    #  4. Generate the Story 
    end_of_story_token_id = tokenizer.convert_tokens_to_ids('<|endofstory|>')

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=500,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        eos_token_id=end_of_story_token_id
    )

    #  5. Decode and Print the Story 
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)

    # Clean up the output by removing the special end token
    story = generated_text.replace('<|endofstory|>', '').strip()
    
    print("\n📜 AI's Story:")
    print(story)
    print("-" * 100)
