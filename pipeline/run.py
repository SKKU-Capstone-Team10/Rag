from peft import PeftModel
from dateutil import parser
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import datetime

class myFAISS(FAISS): ## Dense retrieval
    @classmethod
    def from_texts(cls, keys, values, embedding, metadatas=None, ids=None, **kwargs):
        embeddings = embedding.embed_documents(keys)
        return cls._FAISS__from(values, embeddings, embedding, metadatas=metadatas, ids=ids, **kwargs)

    def add_texts(self, keys, values, metadatas=None, ids=None, **kwargs):
        embeddings = [self.embedding_function(k) for k in keys]
        return self._FAISS__add(values, embeddings, metadatas=metadatas, ids=ids)

class MyProcessor(nn.Module):
    def __init__(self, embedding_model_path, faiss_config):
        super(MyProcessor, self).__init__()
        self.embed = HuggingFaceEmbeddings(model_name=embedding_model_path, model_kwargs={'device': 'cpu'},encode_kwargs={"normalize_embeddings": True})
        self.faiss_config = faiss_config
        self.faiss_retriever = {}
        for db, db_cfg in self.faiss_config.items():
            try:
                vector_db = myFAISS.load_local(db_cfg['INDEX_PATH'], self.embed, db_cfg['INDEX_NAME'])
                print(f"[INFO] Loaded FAISS DB: {db}")
            except Exception as e:
                print(f"[ERROR] {e}")
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
prompt = """Reference materials are as follows:
{docs}
As a personal knowledge assistant, please answer the following question based on the above reference materials. Your response must be entirely grounded in the provided content, and no fabricated information is allowed.
The question is: {query}. Support your answer with examples based on the relevant data.
"""

        return prompt.format(docs=doc, query=query)

    def concat_docs(self, docs):
        if not docs:
            return "[!] 관련 문서를 찾지 못했습니다."
        return "\n\n".join([f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(docs)])

    def process(self, query, database, topk, model, tokenizer, method):
        query = self.checkDate(query)
        print(f"[INFO] Query: {query}")

        docs = self.faiss_retriever[database].similarity_search_with_score(query, k=topk)
        docs = [doc[0].page_content for doc in docs]
        knowledge = self.concat_docs(docs)

        if method == '3':
            return knowledge, knowledge
        else:
            user_prompt = self.get_prompt(knowledge, query)
            response, _ = model.chat(tokenizer, user_prompt, history=[])
            return response, knowledge

def getModel(base_model_path, lora_ckpt_path, embedding_model_path, faiss_config, device):
    runner = MyProcessor(embedding_model_path, faiss_config)
    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_ckpt_path).to(device)
    model.eval()
    return model, tokenizer, runner

def main(base_model_path, lora_ckpt_path, embedding_model_path):
    faiss_config = {
        "news": {"INDEX_PATH": "data/database_sample/news", "INDEX_NAME": "news_index"},
        "reports": {"INDEX_PATH": "data/database_sample/reports", "INDEX_NAME": "reports_index"},
        "prices": {"INDEX_PATH": "data/database_sample/prices", "INDEX_NAME": "prices_index"}
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, runner = getModel(base_model_path, lora_ckpt_path, embedding_model_path, faiss_config, device)

    db_options = {"1": "reports", "2": "prices", "3": "news"}
    method_options = {"1": "1", "2": "2", "3": "3"}

    while True:
        print("\n[질문 입력] (종료하려면 'exit'):")
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        print("\n[데이터베이스 선택] 1: 리포트, 2: 가격, 3: 뉴스")
        db_choice = input(">> ").strip()

        print("\n[질의 방식 선택] 1: 단일 지식, 2: 통합 지식, 3: 문서만 조회")
        method_choice = input(">> ").strip()

        try:
            response, knowledge = runner.process(
                query=query,
                database=db_options[db_choice],
                topk=5, ## 몇개의 문서를 검색할지
                model=model,
                tokenizer=tokenizer,
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
