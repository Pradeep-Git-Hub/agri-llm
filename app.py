
import streamlit as st
import json
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================
# Configuration
# =========================

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "./agri_llm_lora_final"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

THRESHOLD = 0.5


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="AgriLLM",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 AgriLLM")
st.subheader("Crop Disease Knowledge Assistant")

st.write(
    "Ask questions about crop diseases using a "
    "RAG + LoRA powered language model."
)


# =========================
# Load Knowledge Base
# =========================

@st.cache_data
def load_documents():

    with open(
        "./data/knowledge_base.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


documents = load_documents()


# =========================
# Load Embedding Model
# =========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedding_model = load_embedding_model()


# =========================
# Create FAISS Index
# =========================

@st.cache_resource
def create_faiss_index(documents):

    texts = [
        f"{doc['crop']} {doc['disease']} {doc['content']}"
        for doc in documents
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index


index = create_faiss_index(documents)


# =========================
# Load Qwen + LoRA
# =========================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_llm()


# =========================
# Retrieval
# =========================

def retrieve_documents(
    query,
    k=3,
    threshold=THRESHOLD
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if score >= threshold:

            results.append({
                "score": float(score),
                "document": documents[idx]
            })

    return results


# =========================
# Build Context
# =========================

def build_context(results):

    context_parts = []

    for result in results:

        doc = result["document"]

        context_parts.append(
            f"Crop: {doc['crop']}\n"
            f"Disease: {doc['disease']}\n"
            f"{doc['content']}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# =========================
# Generate Answer
# =========================

def generate_answer(query, context):

    prompt = f"""
You are AgriLLM, a crop disease knowledge assistant.

Answer the user's question using ONLY the
provided knowledge base context.

If the answer cannot be supported by the context,
say:

"I don't have enough information in my knowledge base to answer that."

Do not invent facts.

Knowledge Base Context:
{context}

User Question:
{query}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.8,
            do_sample=True
        )

    generated_tokens = outputs[
        0
    ][inputs.input_ids.shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# =========================
# User Interface
# =========================

question = st.text_input(
    "Ask a crop disease question:",
    placeholder="Example: What are the symptoms of tomato early blight?"
)


if st.button("🔍 Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching knowledge base and generating answer..."
        ):

            results = retrieve_documents(
                question
            )

            if not results:

                answer = (
                    "I don't have enough information "
                    "in my knowledge base to answer that."
                )

            else:

                context = build_context(
                    results
                )

                answer = generate_answer(
                    question,
                    context
                )

        st.markdown("### 🤖 Answer")

        st.write(answer)

        if results:

            st.markdown(
                "### 📚 Retrieved Knowledge"
            )

            for result in results:

                doc = result["document"]

                with st.expander(
                    f"{doc['crop']} — "
                    f"{doc['disease']} "
                    f"(similarity: {result['score']:.3f})"
                ):

                    st.write(
                        doc["content"]
                    )
