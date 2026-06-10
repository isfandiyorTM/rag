from __future__ import annotations

import functools
from typing import List, Dict, Tuple

from groq import Groq

from .vector_store import VectorStore


SYSTEM_PROMPT = """You are a helpful, professional FAQ assistant for TechCorp Analytics Solutions.
Your role is to answer questions accurately using ONLY the provided knowledge base context.

Guidelines:
- Answer concisely and professionally.
- If the answer is clearly in the context, provide it directly.
- If the context is insufficient or irrelevant, say: "I don't have enough information in the knowledge base to answer that question. Please contact our support team."
- Never fabricate facts, prices, or details not present in the context.
- Format responses using markdown when helpful (bullet points, bold for key terms).
- Keep answers focused and avoid unnecessary repetition."""


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


def query(
    question: str,
    vector_store: VectorStore,
    conversation_history: List[Dict],
    api_key: str,
    top_k: int = 4,
) -> Tuple[str, List[Tuple[Dict, float]]]:
    query_embedding = get_embeddings([question])[0]
    results = vector_store.search(query_embedding, top_k=top_k)
    context = build_context(results)
    answer = generate_answer(question, context, conversation_history, api_key)
    return answer, results
