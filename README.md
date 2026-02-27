# ai-restaurant-genie
your AI genie for restaurant ideas
# 🍽️ AI Restaurant Branding Generator

A **Streamlit app** that generates **unique restaurant ideas, names, cuisine style, concept, and tagline** using AI and RAG (Retrieval-Augmented Generation).  
It uses a combination of **Groq LLM** for text generation and **HuggingFace embeddings + Chroma** for knowledge retrieval from cuisine context.

---

## **Demo**

- Users can type the type of restaurant they want.
- The AI uses predefined cuisine knowledge to generate:
  1. Unique Restaurant Name  
  2. Cuisine Style  
  3. Concept Description  
  4. Catchy Tagline  

Example:

| User Input                     | AI Output |
|--------------------------------|-----------|
| "Cozy Italian pasta bistro"     | Name: `Bella Tavola`<br>Cuisine: Italian<br>Concept: Cozy pasta bistro with authentic Italian flavors and local ingredients.<br>Tagline: `Taste Italy, One Bite at a Time` |

---

## **Features**

- **Interactive Chat UI** built with Streamlit  
- **Groq LLM** integration for generating restaurant concepts  
- **RAG (Retrieval-Augmented Generation)** using Chroma + HuggingFace embeddings  
- **Conversation Memory** using Streamlit session state  
- Supports multiple cuisine types: Italian, Indian, Japanese, Mexican, French, Thai, Chinese, Greek  

---

## **Project Structure**
restaurant_ai_app/
├─ app.py # Main Streamlit app
├─ requirements.txt # Python dependencies
├─ restaurant_db/ # Optional prebuilt vector store
├─ .env.example # Example of API key variables
└─ .gitignore # Ignore virtualenv, .env, cache files



## **Installation / Setup**

1. **Clone the repository**


git clone https://github.com/yourusername/restaurant-ai-app.git
cd restaurant-ai-app

2.Create a virtual environment

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

3.Install dependencies

pip install -r requirements.txt

Create .env file with your API key

GROQ_API_KEY=your_groq_api_key_here
