from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from core.config import settings

category = ["chart", "general"]
# Template
template_category = "Distinguish the {question} category among {category}. Only respone with a word."
template_chart = '''
Your response must be in json with keys - ticker, start_date, end_date, info.
info field should be list of tuple like (date, information).
Parse the {question} and response with proper keys.
'''
template_general = "Answer the question: Question: {question}"

# Prompt
prompt = ChatPromptTemplate.from_template(template_general)

prompt_category = ChatPromptTemplate.from_template(template_category)
prompt_chart = ChatPromptTemplate.from_template(template_chart)
prompt_general = ChatPromptTemplate.from_template(template_general)

# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=settings.OPENAI_API_KEY)
# model_category = ...
# model_chart = ...
# model_general  = ...
# model_answer = ...

# Chain
chain = (
    {'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

category_chain = (
    {'question': RunnablePassthrough(), 'category': lambda _: category}
    | prompt_category
    | model
    | StrOutputParser()
)

def route_prompt(category: str) -> ChatPromptTemplate:
    if category.strip().lower() == "chart":
        return prompt_chart
    else:
        return prompt_general

final_chain = (
    {'question': RunnablePassthrough()}
    | {
        'question': lambda x: x,
        'category': category_chain
    }
    | RunnableLambda(lambda inputs: {
        "question": inputs["question"],
        "prompt": route_prompt(inputs["category"])
    })
    | (lambda d: d["prompt"].invoke({"question": d["question"]}))
    | model
    | StrOutputParser()
)