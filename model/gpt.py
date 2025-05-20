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

template_ticker = (
    "Determine whether the following question is related to stocks, and extract the stock ticker symbol from the question.\n"
    "Input: {question}\n\n"
    "Rules:\n"
    "1. If the question contains a 1–5 character uppercase English ticker pattern (A–Z), output that ticker only.\n"
    "2. If it includes financial keywords like “stock price”, “earnings”, or “dividend” but no ticker pattern, output `False` only.\n"
    "3. If multiple tickers appear, output the one most relevant to the question’s intent.\n"
    "4. Output exactly one token—either the ticker (e.g., `AAPL`) or `False`—with no additional text or whitespace.\n\n"
    "Output:"
)

prompt = ChatPromptTemplate.from_template(template)
prompt_title = ChatPromptTemplate.from_template(template_title)
prompt_ticker = ChatPromptTemplate.from_template(template_ticker)

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

chain_ticker = (
    {'question': RunnablePassthrough()}
    | prompt_ticker
    | model
    | StrOutputParser()
)