# 🔐 AI-Powered OCR Chatbot with FAISS, PySpark & Keycloak Authentication

An interactive **Streamlit-based web application** that integrates Optical Character Recognition (OCR), document summarization, question-answering, and text generation. It uses **FAISS** for semantic search, **Google Gemini** for LLM capabilities, **Keycloak** for authentication, and **PySpark** for scalable data processing.

---

## 🚀 Features

- 🔐 **User Authentication** using Keycloak (with public demo mode option)
- 🧾 **OCR Extraction** from uploaded image files (JPG, PNG)
- 📚 **Text Summarization** using retrieved documents
- ❓ **Question Answering** over vector-embedded data
- ✍️ **Text Generation** using Google Gemini
- ⚡ **FAISS-powered vector search** over CSV datasets
- 🧠 **Sentence embeddings** via HuggingFace Transformers
- 🎯 Streamlit-based user-friendly UI with responsive sidebar and contact card

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit  
- **LLM & Embeddings:** Google Gemini API, HuggingFace Sentence Transformers  
- **Vector Database:** FAISS  
- **OCR Engine:** Tesseract  
- **Authentication:** Keycloak  
- **Backend/Data Processing:** PySpark  
- **Utilities:** PIL (Pillow), Requests, LangChain

---

## 🔧 Setup Instructions (Local)

1. **Clone this repository**


git clone https://github.com/chaubey09/streamlit_demo_ocr_chatbot.git
cd streamlit_demo_ocr_chatbot

2. Install Dependencies
pip install -r requirements.txt

3. Start Keycloak Using Docker
docker run -p 8080:8080 \
  -e KEYCLOAK_ADMIN=admin \
  -e KEYCLOAK_ADMIN_PASSWORD=admin \
  quay.io/keycloak/keycloak:24.0.2 start-dev

4. Run the Application
streamlit run app.py

