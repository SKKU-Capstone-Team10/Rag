# run.py  ─ 티커별 FAISS 동적 로딩 버전
import os, datetime, dotenv, torch, torch.nn as nn
from dateutil import parser
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API"))

# ────────────────────────────────────────────
# 1) FAISS 래퍼 (변경 없음)
class myFAISS(FAISS):
    @classmethod
    def load_local(cls, folder_path, embeddings, **kwargs):
        return super().load_local(folder_path=folder_path,
                                  embeddings=embeddings, **kwargs)
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, ids=None, **kwargs):
        embeddings = embedding.embed_documents(texts)
        return cls._FAISS__from(texts, embeddings, embedding, metadatas=metadatas, ids=ids, **kwargs)

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        embeddings = [self.embedding_function(t) for t in texts]
        return self._FAISS__add(texts, embeddings, metadatas=metadatas, ids=ids)

# ────────────────────────────────────────────
# 2) Processor
class MyProcessor(nn.Module):
    def __init__(self, embedding_model_path, faiss_base_paths):
        """
        faiss_base_paths: {"news": <folder>, "prices": <folder>}
                          └─ 실제 인덱스는 <folder>/<SYMBOL>/index.faiss
        """
        super().__init__()
        self.embed = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ### [변경] DB별 base 경로만 저장
        self.base_paths = faiss_base_paths

        # ### [변경] {db: {symbol: vector_db}} 캐시 딕셔너리
        self.cache = {db: {} for db in self.base_paths}

    # ───────── 헬퍼 ─────────
    def _load_vector_db(self, db: str, symbol: str):
        """필요 시 디스크에서 로드 후 캐시에 보관"""
        if symbol in self.cache[db]:
            return self.cache[db][symbol]

        idx_path = os.path.join(self.base_paths[db], symbol)
        if not os.path.isdir(idx_path):
            raise FileNotFoundError(f"인덱스 폴더가 없습니다: {idx_path}")

        print(f"[INFO] FAISS 로딩: {idx_path}")
        vectordb = myFAISS.load_local(
            folder_path=idx_path,
            embeddings=self.embed,
            allow_dangerous_deserialization=True,
        )
        self.cache[db][symbol] = vectordb
        return vectordb

    # ───────── 날짜 보강 ─────────
    def check_date(self, query: str) -> str:
        try:
            parser.parse(query, fuzzy=True).date()
            return query
        except Exception:
            now = datetime.datetime.now()
            latest = (now.date() - datetime.timedelta(days=1)
                      if now.hour < 18 else now.date())
            return f"{query} (오늘 날짜: {latest})"

    # ───────── 프롬프트 생성 ─────────
    @staticmethod
    def concat_docs(docs):
        return "\n\n".join(
            f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(docs)
        ) if docs else "[!] 관련 문서를 찾지 못했습니다."

    @staticmethod
    def make_prompt(doc, query):
        return (
            f"""Reference materials are as follows:
            {doc}

            As a personal knowledge assistant, please answer the following question based on the above reference materials. Your response must be entirely grounded in the provided content, and no fabricated information is allowed.

            The question is: {query}. Support your answer with examples based on the relevant data."""
        )

    # ───────── 메인 메서드 ─────────
    def process(self, query, db, symbol, topk, method):
        query = self.check_date(query)
        vectordb = self._load_vector_db(db, symbol)

        # 검색
        hits = vectordb.similarity_search_with_score(query, k=topk)
        docs = [d.page_content for d, _ in hits]
        knowledge = self.concat_docs(docs)

        # method 3 = 문서만
        if method == "3":
            return knowledge, knowledge

        user_prompt = self.make_prompt(knowledge, query)
        comp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that only answers based on provided reference materials."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        answer = comp.choices[0].message.content
        return answer, knowledge


# ────────────────────────────────────────────
# 3) CLI 엔트리
def main(embedding_model_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ### [변경] DB별 ‘베이스 폴더’만 지정
    faiss_base_paths = {
        "news":   os.path.join(BASE_DIR, "news/index"),
        "prices": os.path.join(BASE_DIR, "prices"),
    }

    runner = MyProcessor(embedding_model_path, faiss_base_paths)

    db_opts     = {"1": "prices", "2": "news"}
    method_opts = {"1": "1", "2": "2", "3": "3"}

    while True:
        print("\n[질문 입력] (exit 입력 시 종료):")
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        print("[티커 입력] (AAPL, TSLA 등):")
        symbol = input(">> ").strip().upper()
        if symbol.lower() == "exit":
            break

        print("[데이터베이스 선택] 1: 가격, 2: 뉴스")
        db_key = input(">> ").strip()
        if db_key not in db_opts:
            print("잘못된 입력입니다."); continue

        print("[질의 방식] 1: 단일, 2: 통합, 3: 문서만")
        m_key = input(">> ").strip()
        if m_key not in method_opts:
            print("잘못된 입력입니다."); continue

        try:
            answer, refs = runner.process(
                query=query,
                db=db_opts[db_key],
                symbol=symbol,
                topk=5,
                method=method_opts[m_key],
            )
            print("\n===== [답변] =====\n", answer)
            print("\n===== [검색된 문서] =====\n", refs)
        except Exception as e:
            print(f"[오류] {e}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
