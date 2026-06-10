from __future__ import annotations

import functools
from typing import List, Dict, Tuple

from groq import Groq

from .vector_store import VectorStore


SYSTEM_PROMPT = """You are a helpful, professional FAQ assistant for TechCorp Analytics Solutions.
Answer questions using the provided knowledge base context.

Guidelines:
- Be concise and professional.
- Users often ask in casual or incomplete ways (e.g. "office?" or "price"). Interpret their intent and answer from the context.
- If the context contains relevant information, use it — even if the user's wording is informal or partial.
- Only say you don't know if the context is genuinely unrelated to the question. In that case say: "That topic isn't covered in the knowledge base. For more help, contact support@techcorp.ai"
- Never fabricate facts, prices, or details not present in the context.
- Use markdown (bold, bullet points) to format answers clearly."""


EXPAND_PROMPT = """Rewrite the following user query as a clear, complete question suitable for searching a business FAQ.
Keep it short (one sentence). Return only the rewritten question, nothing else.

User query: {query}"""


@functools.lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    model = _embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def build_context(results: List[Tuple[Dict, float]]) -> str:
    if not results:
        return ""
    sections = []
    for i, (chunk, score) in enumerate(results, 1):
        sections.append(
            f"[Source {i}: {chunk['source']} | Relevance: {score:.0%}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(sections)


def generate_answer(
    query: str,
    context: str,
    conversation_history: List[Dict],
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    client = Groq(api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in conversation_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_message = (
        f"Knowledge Base Context:\n{context}\n\n---\n\nUser Question: {query}"
        if context
        else f"User Question: {query}\n\n(No relevant context found in the knowledge base.)"
    )
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


def _expand_query(question: str, api_key: str) -> str:
    """Rewrite short/casual queries into full questions for better embedding match."""
    if len(question.split()) >= 6:
        return question  # already detailed enough
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": EXPAND_PROMPT.format(query=question)}],
            temperature=0,
            max_tokens=60,
        )
        expanded = resp.choices[0].message.content.strip().strip('"')
        return expanded if expanded else question
    except Exception:
        return question


def query(
    question: str,
    vector_store: VectorStore,
    conversation_history: List[Dict],
    api_key: str,
    top_k: int = 5,
) -> Tuple[str, List[Tuple[Dict, float]]]:
    search_query = _expand_query(question, api_key)
    query_embedding = get_embeddings([search_query])[0]
    results = vector_store.search(query_embedding, top_k=top_k)
    context = build_context(results)
    answer = generate_answer(question, context, conversation_history, api_key)
    return answer, results
