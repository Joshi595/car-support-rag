"""The RAG pipeline as a LangGraph state machine:  condense → retrieve → generate.

- condense:  rewrites a follow-up ("and for EVs?") into a standalone question using the chat history.
- retrieve:  fetches the top-k chunks for that standalone question.
- generate:  answers from the numbered chunks only, citing them as [n].
"""

import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from rag import config
from rag.retrieval import Retriever

REFUSAL = "I'm sorry, I couldn't find that in our documentation. Please contact our customer hotline for help."

# The history goes in as a transcript inside ONE message: given real chat turns, small models
# continue the chat and answer the question instead of rewriting it (measured in eval/).
CONDENSE_PROMPT = """Rewrite the customer's follow-up message as a standalone question for a search engine.
Use the conversation only to resolve references such as "it", "that" or "and for X?".
Do NOT answer the question. Output only the rewritten question.

Example
Conversation:
Customer: How long is the standard warranty?
Assistant: 2 years with unlimited mileage.
Follow-up: what about the battery?
Standalone question: How long is the warranty for the battery?

Conversation:
{history}
Follow-up: {question}
Standalone question:"""

ANSWER_PROMPT = """You are a customer service assistant for a car manufacturer.
Answer the customer's question using ONLY the numbered context passages below.

Rules:
- Cite the passage number for every fact, in square brackets, e.g. [2].
- If the passages do not contain the answer, reply with exactly this sentence and nothing else:
  "{refusal}"
- Never use outside knowledge. Never mention "passages" or "context" to the customer.
- Be concise and friendly; use a short list when listing several items.

Context passages:
{context}"""


class State(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]  # [("user" | "assistant", text), ...]
    k: int
    mode: str
    model: str
    standalone: str
    docs: list[Document]
    answer: str


@lru_cache
def chat_model(name: str) -> ChatOllama:
    # reasoning=False disables the <think> phase on hybrid models: this task needs lookup, not long reasoning.
    return ChatOllama(model=name, temperature=0, reasoning=False, base_url=config.OLLAMA_URL)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def is_refusal(answer: str) -> bool:
    return "couldn't find that in our documentation" in answer.replace("’", "'").lower()


def cited_sources(answer: str, docs: list[Document]) -> list[tuple[int, Document]]:
    """Return the (number, chunk) pairs the answer actually cites, e.g. "[1]" or "[1, 3]"."""
    nums = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in group.split(",")}
    return [(n, docs[n - 1]) for n in sorted(nums) if 1 <= n <= len(docs)]


def build_graph(retriever: Retriever):
    """Compile the graph. Per-request knobs (k, mode, model) travel in the state."""

    def condense(s: State) -> State:
        if not s.get("history"):
            return {"standalone": s["question"]}
        transcript = "\n".join(f"{'Customer' if role == 'user' else 'Assistant'}: {text[:300]}"
                               for role, text in s["history"][-6:])
        prompt = CONDENSE_PROMPT.format(history=transcript, question=s["question"])
        lines = chat_model(s["model"]).invoke([("user", prompt)]).content.strip().splitlines()
        rewritten = lines[0].removeprefix("Standalone question:").strip().strip('"') if lines else ""
        return {"standalone": rewritten or s["question"]}

    def retrieve(s: State) -> State:
        return {"docs": retriever.search(s["standalone"], s["k"], s["mode"])}

    def generate(s: State) -> State:
        # The standalone question already carries the conversation's context, so the answer
        # prompt stays small and grounded in the retrieved passages only.
        system = ANSWER_PROMPT.format(refusal=REFUSAL, context=format_context(s["docs"]))
        msg = chat_model(s["model"]).invoke([("system", system), ("user", s["standalone"])])
        return {"answer": msg.content.strip()}

    g = StateGraph(State)
    g.add_node("condense", condense)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "condense")
    g.add_edge("condense", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def new_state(question: str, history=(), k: int = config.TOP_K, mode: str = "hybrid",
              model: str = config.CHAT_MODEL) -> State:
    return {"question": question, "history": list(history), "k": k, "mode": mode, "model": model}
