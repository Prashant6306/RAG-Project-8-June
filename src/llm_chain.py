from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.utils import format_docs
from src.config import OLLAMA_MODEL_NAME, OLLAMA_URL
from src.constants import PROMPT_TEMPLATE

prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)

llm = ChatOllama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_URL, temperature=0.1)

def build_rag_chain(retriever):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
