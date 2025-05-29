import os
import datetime
import torch
import torch.nn as nn
import dotenv
from dateutil import parser
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")
client = OpenAI(api_key=OPENAI_API_KEY)

class myFAISS(FAISS):
    @classmethod
    def load_local(cls, folder_path, embeddings, **kwargs):
        return super().load_local(folder_path=folder_path, embeddings=embeddings, **kwargs)

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, ids=None, **kwargs):
        embeddings = embedding.embed_documents(texts)
        return cls._FAISS__from(texts, embeddings, embedding, metadatas=metadatas, ids=ids, **kwargs)

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        embeddings = [self.embedding_function(t) for t in texts]
        return self._FAISS__add(texts, embeddings, metadatas=metadatas, ids=ids)

class MyProcessor(nn.Module):
    def __init__(self, embedding_model_path, faiss_config):
        super(MyProcessor, self).__init__()
        self.embed = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.faiss_config = faiss_config
        self.faiss_retriever = {}

        for db, db_cfg in self.faiss_config.items():
            print(f"[DEBUG] Loading '{db}' FAISS index from {db_cfg['INDEX_PATH']}")
            try:
                vector_db = myFAISS.load_local(
                    folder_path=db_cfg['INDEX_PATH'],
                    embeddings=self.embed,
                    allow_dangerous_deserialization=True
                )
                print(f"[INFO] Loaded FAISS DB: {db}")
            except Exception as e:
                print(f"[ERROR] Failed to load FAISS DB '{db}': {e}")
                vector_db = None
            self.faiss_retriever[db] = vector_db

    def checkDate(self, query):
        try:
            parser.parse(query, fuzzy=True).date()
        except:
            now = datetime.datetime.now()
            latest_date = (now.date() - datetime.timedelta(days=1)) if now.hour < 18 else now.date()
            query += f'（오늘 날짜는: {latest_date}）'
        return query

    def get_prompt(self, doc, query):
        return f"""Reference materials are as follows:
{doc}

As a personal knowledge assistant, please answer the following question based on the above reference materials. Your response must be entirely grounded in the provided content, and no fabricated information is allowed.

The question is: {query}. Support your answer with examples based on the relevant data."""

    def concat_docs(self, docs):
        if not docs:
            return "[!] 관련 문서를 찾지 못했습니다."
        return "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(docs)])

    def process(self, query, database, topk, method):
        query = self.checkDate(query)
        print(f"[INFO] Query: {query}")

        if database not in self.faiss_retriever:
            raise ValueError(f"[ERROR] faiss_retriever에 '{database}' 키가 존재하지 않습니다.")
        vector_db = self.faiss_retriever[database]
        if vector_db is None:
            raise ValueError(f"[ERROR] '{database}'에 해당하는 FAISS 인덱스가 로드되지 않았습니다.")

        docs = vector_db.similarity_search_with_score(query, k=topk)
        docs = [doc[0].page_content for doc in docs]
        knowledge = self.concat_docs(docs)

        if method == '3':
            return knowledge, knowledge
        else:
            user_prompt = self.get_prompt(knowledge, query)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that only answers based on provided reference materials."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )
            answer = completion.choices[0].message.content
            return answer, knowledge

def getModel(embedding_model_path, faiss_config, device):
    return MyProcessor(embedding_model_path, faiss_config)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main(embedding_model_path):
    faiss_config = {
        "news": {
            "INDEX_PATH": os.path.join(BASE_DIR, "src/data/database_sample/news"),
            "INDEX_NAME": "news"
        },
        "prices": {
            "INDEX_PATH": os.path.join(BASE_DIR, "src/data/database_sample/prices"),
            "INDEX_NAME": "prices"
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runner = getModel(embedding_model_path, faiss_config, device)

    db_options = {"1": "prices", "2": "news"}
    method_options = {"1": "1", "2": "2", "3": "3"}

    while True:
        print("\n[질문 입력] (종료하려면 'exit'):")
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        print("\n[데이터베이스 선택] 1: 가격, 2: 뉴스")
        db_choice = input(">> ").strip()

        print("\n[질의 방식 선택] 1: 단일 지식, 2: 통합 지식, 3: 문서만 조회")
        method_choice = input(">> ").strip()

        try:
            response, knowledge = runner.process(
                query=query,
                database=db_options[db_choice],
                topk=5,
                method=method_options[method_choice]
            )
            print("\n===== [답변] =====")
            print(response)
            print("\n===== [검색된 문서] =====")
            print(knowledge)
        except Exception as e:
            print(f"[오류] {e}")

if __name__ == '__main__':
    import fire
    fire.Fire(main)
