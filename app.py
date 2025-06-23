import base64
import pytesseract
from PIL import Image
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import streamlit as st
from pyspark.sql import SparkSession
import os
import requests

# Keycloak Configuration
KEYCLOAK_URL = "http://localhost:8080"
REALM = "master"
CLIENT_ID = "anmol111"
TOKEN_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"

# Custom CSS for styled bio card
st.markdown("""
    <style>
    .bio-card { 
        background-color: #ffffff; 
        border-radius: 12px; 
        padding: 16px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
        text-align: center; 
        margin-bottom: 20px; 
    }
    .bio-card img { 
        border-radius: 50%; 
        width: 150px !important; 
        height: 150px; 
        object-fit: cover; 
    }
    .bio-card a { 
        color: #4a90e2; 
        text-decoration: none; 
    }
    .bio-card a:hover { 
        text-decoration: underline; 
    }
    </style>
""", unsafe_allow_html=True)

st.title("Chat Application with OCR Functionality, PySpark, FAISS & Keycloak Authentication")
st.write("Welcome to your AI-powered Chat Application! Please log in to continue.")

# Sidebar for use case selection, login, and credentials
with st.sidebar:
    # Login Section
    st.subheader("Login to Keycloak")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    # Handle Login
    if login_button:
        if username and password:
            payload = {"client_id": CLIENT_ID, "grant_type": "password", "username": username, "password": password}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            try:
                response = requests.post(TOKEN_URL, data=payload, headers=headers)
                response.raise_for_status()
                token_data = response.json()
                st.session_state["access_token"] = token_data.get("access_token")
                st.success("✅ Login Successful!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Login failed: {e}")
        else:
            st.warning("⚠ Please enter username and password.")

    if "access_token" in st.session_state:
        if st.button("Logout"):
            st.session_state.pop("access_token", None)
            st.success("🚪 Logged out successfully!")
            st.stop()

    # Use Case Selection at the top
    use_case = st.selectbox("Choose a Use Case", ["OCR Extraction", "Summarization", "Question and Answer", "Text Generation"])

    # Credentials Bio Section at the end
    st.subheader("Contact me !")
    try:
        profile_pic = Image.open("IMG_0924(1).jpg")
        st.image(profile_pic, width=150, caption="Anmol Chaubey", output_format="PNG")
    except FileNotFoundError:
        st.warning("Profile photo 'IMG_0924(1).jpg' not found. Please add it to the project directory.")
    st.markdown("""
        <div class="bio-card">
            <strong>Anmol Chaubey</strong><br>
            Email: <a href="mailto:anmolchaubey820@gmail.com">anmolchaubey820@gmail.com</a><br>
            <a href="https://www.linkedin.com/in/anmol-chaubey-120b42206/" target="_blank">LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)

Disable this for free deployment
if "access_token" not in st.session_state:
    st.warning("🔑 Please log in to access features.")
    st.stop()

@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("Chat Application") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

spark = get_spark_session()

@st.cache_data
def load_data_and_create_index():
    file_path = "healthcare_dataset.csv"
    df = spark.read.csv(file_path, header=True, inferSchema=True).limit(10000)
    documents = df.toPandas().to_dict(orient="records")
    text_documents = [" ".join(str(v) for v in doc.values()) for doc in documents]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store.as_retriever()

retriever = load_data_and_create_index()

genai.configure(api_key="AIzaSyAAndXw9jGyBNkfnNo70j8rQq4cqwYe-k")
llm = genai.GenerativeModel('gemini-1.5-flash-latest')

if use_case == "OCR Extraction":
    st.header("📄 OCR Text Extraction")
    uploaded_file = st.file_uploader("Upload an image for OCR", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.text_area("Extracted Text:", text, height=200)

elif use_case == "Summarization":
    st.header("📝 Summarization")
    search_query = st.text_input("Enter a Search Query:")
    if st.button("Summarize") and search_query.strip():
        retrieved_docs = retriever.get_relevant_documents(search_query)
        if retrieved_docs:
            combined_content = " ".join([doc.page_content for doc in retrieved_docs])
            response = llm.generate_content(f"Summarize this: {combined_content}")
            st.write(response.text)
        else:
            st.warning("No relevant documents found!")

elif use_case == "Question and Answer":
    st.header("💬 Question and Answer")
    question = st.text_input("Enter your Question:")
    if st.button("Get Answer") and question.strip():
        retrieved_docs = retriever.get_relevant_documents(question)
        if retrieved_docs:
            context = " ".join([doc.page_content for doc in retrieved_docs])
            response = llm.generate_content(f"Answer this based on context: {question}\nContext: {context}")
            st.write(response.text)
        else:
            st.warning("No relevant documents found!")

elif use_case == "Text Generation":
    st.header("🖊 Text Generation")
    prompt = st.text_area("Enter a prompt:")
    if st.button("Generate Text") and prompt.strip():
        response = llm.generate_content(prompt)
        st.write(response.text)

spark.stop()
