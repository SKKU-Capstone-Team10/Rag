import os
import json
import glob
from typing import List, Dict
from tqdm import tqdm

import pandas as pd
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ✅ 1. GPU 지원 임베딩 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": str(device)},
    encode_kwargs={"normalize_embeddings": True},
)

# ------------------------------------------------------------
# ✅ 2. CSV → Document 변환 유틸
# ------------------------------------------------------------
def csv_to_symbol_docs(csv_paths: List[str]) -> Dict[str, List[Document]]:
    """
    csv_paths: CSV 파일 경로들 (ex. ./news/AAPL.csv, ./news/GOOG.csv)
    return    : { "AAPL": [Document, ...], "GOOG": [...], ... }
    """
    symbol_docs: Dict[str, List[Document]] = {}

    for path in csv_paths:
        # 2-1. 심볼 추출 (파일명 → AAPL.csv → AAPL)
        symbol = os.path.splitext(os.path.basename(path))[0]

        df = pd.read_csv(path, encoding="utf-8")
        required = {"title", "published", "descriptionText", "storyPath"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} 필수 컬럼 부족: {required}")

        docs = [
            Document(
                page_content=f"{row.title} {row.descriptionText} "
                             f"{row.storyPath} (published: {row.published})"
            )
            for _, row in df.iterrows()
        ]
        symbol_docs[symbol] = docs

    return symbol_docs

# ────────────────────────────────────────────
# ✅ 3. symbol별 인덱스 → 개별 폴더 저장
def build_and_save_indices(
    csv_paths: List[str],
    base_save_folder: str,
    batch_size: int = 64,
):
    symbol_docs = csv_to_symbol_docs(csv_paths)

    for symbol, docs in symbol_docs.items():
        print(f"[INFO] ▶ {symbol}: 문서 {len(docs)}개")

        raw_texts = [d.page_content for d in docs]
        embeddings = []
        for i in tqdm(range(0, len(raw_texts), batch_size),
                      desc=f"Embedding {symbol}"):
            batch = raw_texts[i:i+batch_size]
            embeddings.extend(embedder.embed_documents(batch))

        vs = FAISS.from_embeddings(
            text_embeddings=list(zip(raw_texts, embeddings)),
            embedding=embedder,
        )

        save_path = os.path.join(base_save_folder, symbol)
        os.makedirs(save_path, exist_ok=True)
        vs.save_local(save_path)
        print(f"[SAVE] {save_path}/index.faiss, index.pkl")

# ✅ 3. 실행
if __name__ == "__main__":
    # 예시: news 폴더 안의 모든 CSV 를 읽어 하나로 묶는다
    news_csv_list   = glob.glob("./data/news/*.csv")
    prices_csv_list = glob.glob("./data/prices/*.csv")

    news_folder = "./data/news/index"
    prices_folder = "./data/prices"

    news_csvs   = glob.glob("./data/news/*.csv")
    prices_csvs = glob.glob("./data/prices/*.csv")

    build_and_save_indices(news_csvs,   news_folder)
    # build_and_save_indices(prices_csvs, "./src/data/database_sample/prices")