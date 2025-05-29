import os
import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ✅ 1. GPU 지원 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",  # 다른 모델도 가능
    model_kwargs={"device": "cuda"},  # ← GPU 사용
    encode_kwargs={"normalize_embeddings": True}
)

# ✅ 2. 벡터 인덱스 생성 및 저장 함수
def build_and_save_index(json_path, save_folder, embedding_model):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "news" in json_path:
        texts = [item["summary"] + " " + item["content"] for item in data]
    elif "prices" in json_path:
        texts = [item["content"] for item in data]
    else:
        texts = [json.dumps(item) for item in data]

    documents = [Document(page_content=text) for text in texts]

    print(f"[INFO] 문서 수: {len(documents)}")

    # ✅ 2-1. tqdm을 사용한 임베딩 진행
    raw_texts = [doc.page_content for doc in documents]
    print("[INFO] 임베딩 생성 중 (GPU 사용)...")
    embeddings = []
    batch_size = 64  # GPU는 보통 64 이상 가능
    for i in tqdm(range(0, len(raw_texts), batch_size), desc="Embedding"):
        batch = raw_texts[i:i + batch_size]
        embeddings.extend(embedding_model.embed_documents(batch))

    # ✅ 2-2. FAISS 객체 생성
    text_embedding_pairs = list(zip(raw_texts, embeddings))

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedding_model
    )

    # ✅ 2-3. 저장
    os.makedirs(save_folder, exist_ok=True)
    vectorstore.save_local(folder_path=save_folder)

    print(f"[INFO] FAISS 인덱스 저장 완료: {save_folder}/index.faiss, index.pkl")

# ✅ 3. 실행
if __name__ == "__main__":
    news_json = "./src/data/database_sample/news/news.json"
    prices_json = "./src/data/database_sample/prices/prices.json"
    news_folder = "./src/data/database_sample/news"
    prices_folder = "./src/data/database_sample/prices"

    build_and_save_index(news_json, news_folder, embedding_model)
    build_and_save_index(prices_json, prices_folder, embedding_model)
