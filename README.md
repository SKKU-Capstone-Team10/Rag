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

### ⚙️ Environment Variables

Create a file named .env in the project root and add the following:
<code><pre>
OPENAI_API_KEY=[Your OpenAI API Key Here]
APIFY_API_KEY=[Your Apify API Key Here]
</code></pre>

Get your API keys from:<br>
	• https://platform.openai.com/api-keys<br>
	• https://console.apify.com/settings/integrations

---

### 📦 Usage

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

	• Trained model and LoRA adapter are saved in ../model/.

---

📌 3. Run FastAPI Server
```bash
cd ..
uvicorn main:app --port 8001
```


---

### 📁 Project Structure

🔹 Top-Level Files

| File                   | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `main.py`              | **Main entry point** – FastAPI app controller and page router           |
| `requirements.txt`     | Python dependency list (pip)                                            |
| `.env`                 | You must create it.                                                     |


---

🔸 Modules & Subdirectories

| Folder              | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `api/`              | API routers, End Points are defined here                                |
| `core/`             | Loading `.env`                                                          |
| `data/`             | Stock News / Retrieval Model                                            |
| `model/`            | Porjection Layer                                                        |
| `schemas/`          | Req & Res data field definitions                                        |


---


<div align="center">
  <i>Capstone Team 10 · Sungkyunkwan University</i>
</div>



