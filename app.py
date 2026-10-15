"""Streamlit chat UI for the RAG assistant.   Run:  streamlit run app.py"""

import ollama
import streamlit as st

from rag import config
from rag.graph import build_graph, cited_sources, is_refusal, new_state
from rag.retrieval import MODES, load_retriever

st.set_page_config(page_title="Car Support Assistant", page_icon="🚗")


@st.cache_resource(show_spinner="Loading knowledge base index…")
def get_graph():
    return build_graph(load_retriever())


@st.cache_data(ttl=60)
def chat_models() -> list[str]:
    """Chat models pulled in the local Ollama. Embedding and thinking models are filtered out:
    some thinking builds (e.g. qwen3:4b-2507) can't switch thinking off and take ~50s per answer."""
    try:
        client = ollama.Client(host=config.OLLAMA_URL)
        names = {m.model for m in client.list().models
                 if "completion" in (caps := client.show(m.model).capabilities or []) and "thinking" not in caps}
    except Exception:
        names = set()
    return sorted(names | {config.CHAT_MODEL})


def render_sources(msg: dict) -> None:
    """Show the rewritten search query (for follow-ups) and the sources the answer cites."""
    if msg.get("standalone") and msg["standalone"] != msg.get("question"):
        st.caption(f"🔎 Searched for: *{msg['standalone']}*")
    if not msg.get("sources"):
        return
    label = "Sources" if msg["cited"] else "Retrieved (no inline citations)"
    st.markdown(f"**{label}:** " + " · ".join(
        f"[{n}] {d.metadata['title']} › {d.metadata['section']}" for n, d in msg["sources"]))
    with st.expander("Show source passages"):
        for n, d in msg["sources"]:
            st.markdown(f"**[{n}] {d.metadata['title']} › {d.metadata['section']}** — `{d.metadata['source']}.txt`")
            st.text(d.page_content.split("\n", 1)[1])


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K chunks", 1, 10, config.TOP_K, help="How many chunks are passed to the model.")
    mode = st.radio("Retrieval", MODES, horizontal=True,
                    help="hybrid = BM25 keywords + dense vectors, fused with Reciprocal Rank Fusion")
    models = chat_models()
    model = st.selectbox("Chat model", models, index=models.index(config.CHAT_MODEL))
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
    st.divider()
    st.caption(f"Embeddings: `{config.EMBED_MODEL}` · Pipeline: condense → retrieve → generate (LangGraph)")

# ── Chat ─────────────────────────────────────────────────
st.title("🚗 Car Support Assistant")
st.caption("Ask about vehicle features, service, warranty, ordering, electric vehicles or support channels.")

try:
    graph = get_graph()
except Exception as e:  # Ollama down or models missing: tell the user how to fix it
    st.error(f"Could not build the index via Ollama at {config.OLLAMA_URL}: {e}\n\n"
             f"Start Ollama and run `ollama pull {config.EMBED_MODEL}` and `ollama pull {config.CHAT_MODEL}`.")
    st.stop()

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m)

if question := st.chat_input("e.g. How long is the battery warranty?"):
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        final: dict = {}

        def tokens():
            """Stream only the generate node's tokens; collect the final state on the side."""
            for kind, data in graph.stream(new_state(question, history, k, mode, model),
                                           stream_mode=["messages", "values"]):
                if kind == "messages" and data[1].get("langgraph_node") == "generate":
                    yield data[0].content
                elif kind == "values":
                    final.update(data)

        try:
            st.write_stream(tokens())
        except Exception as e:
            st.error(f"Model call failed ({e}). Is `{model}` pulled in Ollama?")
            st.stop()

        answer, docs = final["answer"], final["docs"]
        cited = cited_sources(answer, docs)
        msg = {"role": "assistant", "content": answer, "question": question, "standalone": final["standalone"],
               "cited": bool(cited),
               # fall back to showing what was retrieved if the model forgot to cite (but not for refusals)
               "sources": cited or ([] if is_refusal(answer) else list(enumerate(docs, 1)))}
        render_sources(msg)
    st.session_state.messages.append(msg)
