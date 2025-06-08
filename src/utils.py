import re

def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)