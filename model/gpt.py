from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.config import settings

# Prompt
template = "Answer the question: Question: {question}"

prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=settings.OPENAI_API_KEY)

chain = (
    {'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)