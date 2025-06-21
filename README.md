<div align="center">
  <h1>🚀 Capstone RAG AI Server</h1>
  <p>Streamlit‑powered AI chatbot &amp; data‑viz web application</p>
</div>

<br><hr><br>

<h2>🛠️ Setup</h2>

<pre><code># 1 – Create &amp; activate virtual env
Python version 3.12.7
python3 -m venv [name]
source [name]/bin/activate

# 2 – Install dependencies
pip install -r requirements.txt
</code></pre>

<br><hr><br>

<h2>⚙️ Environment Variable</h2>
Create a environment vairable named <code>.env</code><br>
Fill the content like below<br>
Get your API keys on below URLs.<br>
<a href=https://platform.openai.com/api-keys target="_blank">OpenAI API Key</a><br>
<a href=https://console.apify.com/settings/integrations target="_blank">APIFY API Key</a>
<pre><code>
  OPENAI_API_KEY=[Your OpenAI API Key Here]
  APIFY_API_KEY=[Your APIFY API Key Here]
</code></pre>

<h2>⚡ Run</h2>

<pre><code>uvicorn main:app --port 8001</code></pre>

<br><hr><br>
## 📁 Project Structure

### 🔹 Top-Level Files

| File                   | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `main.py`              | **Main entry point** – FastAPI app controller and page router           |
| `requirements.txt`     | Python dependency list (pip)                                            |
| `.env`                 | You must create it.                                                     |

---

### 🔸 Modules & Subdirectories

| Folder              | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `api/`              | API routers, End Points are defined here                                |
| `core/`             | Loading `.env`                                                          |
| `data/`             | Stock News / Retrieval Model                                            |
| `model/`            | Porjection Layer                                                        |
| `schemas/`          | Req & Res data field definitions                                        |

---

<div align="center">
  <i>Capstone Team 10 · Sungkyunkwan University</i>
</div>
