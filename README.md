<div align="center">
  <h1>🚀 Capstone RAG AI Server</h1>
  <p>Streamlit-powered AI chatbot & data-visualization web application</p>
</div>

---

## 🛠️ Setup

### 1. Clone Repository & Set Up Environment

```bash
git clone https://github.com/SKKU-Capstone-Team10/Rag.git
cd Rag

# Python version: 3.12.7
python3 -m venv rag
source rag/bin/activate

pip install -r requirements.txt
```

---

⚙️ Environment Variables

Create a file named .env in the project root and add the following:

OPENAI_API_KEY=[Your OpenAI API Key Here]
APIFY_API_KEY=[Your Apify API Key Here]

Get your API keys from:
	•	https://platform.openai.com/api-keys
	•	https://console.apify.com/settings/integrations

---

📦 Usage

📌 1. Process Embeddings
```bash
cd data/
python process_embedding.py
```

---

📌 2. Train Embedding Model
```bash
cd ../train_model/
python train.py
```

	•	Trained model and LoRA adapter are saved in ../model/.

---

📌 3. Run FastAPI Server
```bash
cd ..
uvicorn main:app --port 8001
```


---

📁 Project Structure

🔹 Top-Level Files

File	Purpose
main.py	FastAPI entry point and route definitions
requirements.txt	Python dependencies
.env	Environment variables file


---

🔸 Modules & Subdirectories

Folder	Purpose
api/	FastAPI endpoints
core/	Environment loading and configuration
data/	News and financial data, embedding scripts
model/	Trained models and projection layers
schemas/	Request/response schema (Pydantic)
train_model/	Triplet-based embedding training with LoRA


---


<div align="center">
  <i>Capstone Team 10 · Sungkyunkwan University</i>
</div>



