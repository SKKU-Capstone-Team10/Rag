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

template_events = (
    "Your task is to generate a list of pairs of [timestamp] and [event] "
    "from given csv file following the [Rules].\n"
    "csv file: {csv}\n\n"

    "[Rules]:\n"
    "0. Generate every contents in English.\n"
    "1. [timestamp] must be UTC time format like 2025-05-23 12:22:22\n"
    "2. The columns published is UNIX time and **not directly related** with the [timestamp] to be generated.\n"
    "3. [event] must be short than 5 words.\n\n"

    "Output **only** the Python literal list of tuples (no additional text or notes).\n"
    "Example:\n"
    "[('2025-05-24 06:15:00', 'iPhone tariff threat'), ('2025-05-24 06:40:00', 'Cook calls governor'), ('2025-05-24 11:00:00', 'Ive joins OpenAI'), ('2025-05-24 17:00:00', 'Production unfeasible'), ('2025-05-24 21:30:00', 'Worst BigTech performer'), ('2025-05-25 16:00:00', 'OpenAI buys Ive startup'), ('2025-05-25 16:00:00', 'Trump tariff threat')]"
)

prompt = ChatPromptTemplate.from_template(template)
prompt_title = ChatPromptTemplate.from_template(template_title)
prompt_events = ChatPromptTemplate.from_template(template_events)

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

chain_events = (
    {'csv': RunnablePassthrough()}
    | prompt_events
    | model
    | StrOutputParser()
)