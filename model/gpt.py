from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.config import settings

# Prompt
template = "Answer the question: Question: {question}"

template_title = (
    "Read the user's question and generate a concise, descriptive title that represents it.\n"
    "Question: {question}\n"
    "Title:"
)

prompt = ChatPromptTemplate.from_template(template)
prompt_title = ChatPromptTemplate.from_template(template_title)

# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=settings.OPENAI_API_KEY)

# Answer the query
chain = (
    {'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Create a title for the query
chain_title = (
    {'question': RunnablePassthrough()}
    | prompt_title
    | model
    | StrOutputParser()
)