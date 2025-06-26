import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TFIDFRetriever, EnsembleRetriever
import torch
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import os
import pickle



st.set_page_config(page_title="Asisten Akademik Informatika", layout="wide")

# === Load Retriever ===
@st.cache_resource
def load_retriever():

    # Load dokumen
    with open("Dokumen/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore_mpnet = FAISS.load_local("faiss_mpnet_index", embedding_model, allow_dangerous_deserialization= True)
    cosine_retriever_mpnet = vectorstore_mpnet.as_retriever(search_type="similarity", k=5)

    # Load kembali TFIDF
    tfidf_retriever = TFIDFRetriever.from_documents(documents)
    tfidf_retriever.k = 5

    # Buat kembali hybrid retriever
    hybrid_retriever_mpnet = EnsembleRetriever(
        retrievers=[tfidf_retriever, cosine_retriever_mpnet],
        weights=[0.5, 0.5]
    )
    
    return hybrid_retriever_mpnet

# === Load LLaMA 3.2 + LoRA ===
@st.cache_resource
def load_llm():

    HF_TOKEN = "huggingface token"
    login(token=HF_TOKEN)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        trust_remote_code=True
    )

    # === Load model dengan adapter LoRA ===
    model = PeftModel.from_pretrained(base_model, "llama3-finetune-chatbot")

    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return llm_pipeline, model


# === Initialize ===
retriever = load_retriever()
llm_pipeline, model_instance = load_llm()


# === Streamlit UI ===
st.title("Asisten Akademik Teknik Informatika ITS (RAG + Finetune)")
st.markdown("Chatbot ini menggunakan hybrid retriever (TF-IDF + MPNet) dan Finetune (LLaMA 3.2) untuk menjawab pertanyaan berdasarkan dokumen silabus.")

st.markdown(f"**Model Device**: `{next(model_instance.parameters()).device}`")


# === Input User ===
user_input = st.text_area("Masukkan pertanyaan:", placeholder="Contoh: Berapa SKS untuk mata kuliah Data Mining?")
submit = st.button("Tanyakan")

# === RAG Function ===
def rag_hybrid_mpnet_llama(query: str):
    retrieved_docs = retriever.get_relevant_documents(query)
    combined_context = " ".join(doc.page_content for doc in retrieved_docs)

    if not any(word in combined_context.lower() for word in query.lower().split()):
        return None, None, retrieved_docs


    prompt = f"Context: {combined_context}\nQuestion: {query}\nAnswer:"
    result = llm_pipeline(prompt,max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)

    # Full Answr
    out = result[0]['generated_text']
    
    # Answer
    answer = out.split("Answer:")[-1]

    sen_answer = answer.split(".")

    return answer, out, retrieved_docs

# === Handle Output ===
if submit and user_input.strip():
    with st.spinner("Mengambil informasi dan menjawab..."):
        answer, context, docs_used = rag_hybrid_mpnet_llama(user_input.strip())

        if answer is None:
            st.warning("Maaf, tidak ada informasi yang relevan di dalam dokumen untuk menjawab pertanyaan ini.")
        else:
            st.markdown("## Jawaban")
            st.success(answer)  

            with st.expander("Context (klik untuk melihat)"):
                st.write(context)

            st.markdown("## Dokumen Terkait")
            for i, doc in enumerate(docs_used):
                with st.expander(f"Dokumen {i+1}"):
                    st.write(doc.page_content.strip())

