# ✨ Text Generation with GPT-2

This project is part of my **Prodigy InfoTech internship**. It demonstrates how to fine-tune a pre-trained **GPT-2 model** on a custom dataset (`stories.txt`) to generate new, original short stories.

---

## 📖 Project Overview

The goal of this project is to adapt the powerful GPT-2 language model for a specific creative task: **story generation**.  
By training it on a dataset of stories, the model learns the **style, structure, and vocabulary** of the text, enabling it to generate **new, coherent story snippets** from a user prompt.

The project is split into two main scripts:

- **`train_final.py`** → Handles the fine-tuning process.  
  Loads the base GPT-2 model, prepares the `stories.txt` dataset, trains the model, and saves it.  

- **`generate_final.py`** → Loads the fine-tuned model and provides an **interactive CLI** for generating stories based on user prompts.  

---

## 🛠️ Technology Stack

- Python  
- PyTorch  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) & [Datasets](https://huggingface.co/docs/datasets/index)  
- Google Colab *(Recommended for GPU acceleration)*  

---

## 🚀 Setup & Run on Google Colab (Recommended)

Using **Google Colab** is the easiest way to run this project since it provides a **free GPU** for faster training.

### Step 1: Upload Project to Google Drive
1. Ensure all files (`train_final.py`, `generate_final.py`, `stories.txt`) are inside a folder named **`textgen/`**.  
2. Open Google Drive.  
3. Drag and drop the `textgen` folder into **My Drive**.  

### Step 2: Set Up the Colab Notebook
1. Open [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.  
2. **Enable GPU**:  
   Go to `Runtime → Change runtime type` → select **T4 GPU**.  
3. **Mount Google Drive** (run in first cell):  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
4. **Install dependencies:**
   ```python
   !pip install torch transformers datasets

### Step 3: Train the Model

Run the training script (adjust path if needed):
   ```python
   !python "/content/drive/My Drive/textgen/train_final.py"
   ```
### Step 4: Generate a Story 

After training, run:
   ```python
   !python "/content/drive/My Drive/textgen/generate_final.py"
   ```
The script will prompt you for input. Type your story starter and press Enter.

## 📂 File Structure
  
```markdown
textgen/
├── train_final.py    
├── generate_final.py 
├── stories.txt        
└── results_final/
    └── final_model/   
