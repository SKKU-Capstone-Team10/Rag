<div align="center">
  <h1>🚀 Capstone RAG AI Server</h1>
  <p>Streamlit-powered AI chatbot & data-visualization web application</p>
</div>

---

## Setup

### 1. Clone Repository & Set Up Environment

```bash
git clone https://github.com/SKKU-Capstone-Team10/Rag.git
cd Rag

# Python version: 3.12.7
python3 -m venv rag
source rag/bin/activate

pip install -r requirements.txt


⸻

⚙️ Environment Variables

Create a file named .env in the project root and add the following keys:

Obtain API keys from:
	•	OpenAI API Key
	•	Apify API Key

OPENAI_API_KEY=[Your OpenAI API Key Here]
APIFY_API_KEY=[Your Apify API Key Here]


⸻

Usage

1. Process Embeddings

Generate dense vector embeddings for retrieval:

cd data/
python process_embedding.py


⸻

2. Train Embedding Model

Train a Bi-Encoder model using triplet loss and apply LoRA fine-tuning:

cd ../train_model/
python train.py

	•	The trained model and LoRA adapter will be saved in the ../model/ directory.

⸻

3. Run FastAPI Server

Launch the API server:

cd ..
uvicorn main:app --port 8001


⸻

📁 Project Structure

Top-Level Files

File	Purpose
main.py	FastAPI entry point and route definitions
requirements.txt	List of required Python packages
.env	Environment variable configuration file


⸻

🔸 Modules & Subdirectories

Folder	Purpose
api/	FastAPI endpoint definitions
core/	Environment variable loading and global configuration
data/	News and financial dataset handling, embedding generation scripts
model/	Trained model weights and projection layer storage
schemas/	Request/response schema definitions using Pydantic
train_model/	Triplet-based embedding model training and LoRA fine-tuning logic


⸻


<div align="center">
  <i>Capstone Team 10 · Sungkyunkwan University</i>
</div>
```
