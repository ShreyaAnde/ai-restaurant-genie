# app.py
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from pydantic import Field

# -----------------------
# Load API Key
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="AI Restaurant Generator", page_icon="🍽️")
st.title("🍽️ AI Restaurant Branding Generator")
st.write("Generate unique restaurant ideas using AI + RAG")

# -----------------------
# Groq LLM Wrapper (Pydantic-safe)
# -----------------------
from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)

class GroqLLM(LLM):
    """LangChain wrapper for Groq Chat LLM"""

    client: object = Field(..., description="Groq client instance")
    model_name: str = Field(default="openai/gpt-oss-20b", description="Groq model name")
    temperature: float = Field(default=0.7, description="Temperature for generation")

    class Config:
        arbitrary_types_allowed = True  # allow client to be any object

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None) -> str:
        """Call Groq Chat Completion API and return text"""
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a restaurant branding expert."},
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

# Initialize LLM
llm = GroqLLM(
    client=groq_client,
    model_name="openai/gpt-oss-20b",
    temperature=0.7
)

# -----------------------
# Embedding Model
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# Cuisine Knowledge Base
# -----------------------
cuisine_docs = [
    "Italian cuisine includes pasta, pizza, risotto, olive oil and Mediterranean flavors.",
    "Indian cuisine is rich in spices like turmeric, cumin, coriander and garam masala.",
    "Japanese cuisine includes sushi, ramen, tempura and umami flavors.",
    "Mexican cuisine includes tacos, burritos, beans, corn and chili peppers.",
    "French cuisine includes fine dining dishes, pastries, cheese and wine.",
    "Thai cuisine balances sweet, sour, salty and spicy flavors.",
    "Chinese cuisine includes dumplings, noodles and stir-fried dishes.",
    "Greek cuisine includes feta cheese, olives, grilled meats and fresh vegetables."
]

# -----------------------
# Cache Vector Store
# -----------------------
@st.cache_resource
def load_vectorstore():
    return Chroma.from_texts(
        cuisine_docs,
        embedding=embeddings,
        persist_directory="./restaurant_db"
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------
# Session Memory
# -----------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# -----------------------
# Custom Prompt
# -----------------------
template = """
You are a restaurant branding expert.

Using the cuisine context below, generate:

1. Unique Restaurant Name
2. Cuisine Style
3. Concept Description
4. Catchy Tagline

Context:
{context}

User Request:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# -----------------------
# RAG Chain
# -----------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# -----------------------
# Chat UI
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Tell me what type of restaurant you want...")

if user_input:
    st.chat_message("user").write(user_input)
    response = qa_chain.invoke({"question": user_input})
    answer = response["answer"]
    st.chat_message("assistant").write(answer)