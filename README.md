Text Generation with GPT-2 


This project is a task completed for the Prodigy InfoTech internship. It demonstrates how to fine-tune a pre-trained GPT-2 model on a custom dataset (stories.txt) to generate new, original short stories.


📖 Project Overview
The goal of this project is to adapt the powerful GPT-2 language model for a specific creative task: story generation. By training it on a dedicated dataset of stories, the model learns the style, structure, and vocabulary of the provided text, enabling it to generate new, coherent story snippets from a user prompt.
The project is split into two main Python scripts:
train_final.py: This script handles the fine-tuning process. It loads the base GPT-2 model, prepares the custom stories.txt dataset, and trains the model. The newly trained, specialized model is then saved.
generate_final.py: This script loads the fine-tuned model and provides an interactive command-line interface for users to enter a prompt and receive a generated story in response.


🛠️ Technology Stack
Python
PyTorch
Hugging Face transformers & datasets Libraries
Google Colab (Recommended for GPU acceleration)



🚀 Setup & Run on Google Colab (Recommended)
Using Google Colab is the easiest way to run this project, as it provides a free GPU for faster training.


Step 1: Upload Your Project to Google Drive
On your computer, ensure all your project files (train_final.py, generate_final.py, stories.txt) are inside a single folder named textgen.
Open your Google Drive.
Drag and drop the entire textgen folder into the main "My Drive" section.


Step 2: Set Up the Colab Notebook
Go to Google Colab and create a New Notebook.
Enable the GPU: Go to Runtime -> Change runtime type and select T4 GPU from the "Hardware accelerator" dropdown.
Mount Your Google Drive: In the first code cell, run the following to connect the notebook to your Google Drive files. You may need to authorize access.

[from google.colab import drive
drive.mount('/content/drive')]


Install Libraries: In the next cell, install the necessary Python packages.

[!pip install torch transformers datasets]


Step 3: Train the Model
In a new cell, run the training script. The command points to the path where your file is located in Google Drive.

[!python "/content/drive/My Drive/textgen/train_final.py"]


Step 4: Generate a Story

After training is complete, run the generation script in a new cell to start creating stories.

[!python "/content/drive/My Drive/textgen/generate_final.py"]


The script will prompt you for input directly in the cell's output. Type your story starter and press Enter.


📂 File Structure


.
└── textgen/

    ├── train_final.py                                                
    ├── generate_final.py                                         
    ├── stories.txt                                                     
    └── results_final/
      
        └── final_model/   
