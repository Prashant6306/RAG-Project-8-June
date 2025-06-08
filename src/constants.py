DEFAULT_QUERY = "Where did the Minoans live?"

PROMPT_TEMPLATE = """
Human: You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer, say you don't know. Don't make up an answer.

<context>
{context}
</context>

<question>
{question}
</question>

Assistant:"""

