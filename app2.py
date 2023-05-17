import streamlit as st
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
# Import load summarize chain
from langchain.chains.summarize import load_summarize_chain

def constitution():
    load_dotenv()
    pdf_reader = PdfReader("/Users/gabrielrenno/Documents/ASK_PDF/CF88_Livro_EC91_2016.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    
    chunks = text_splitter.split_text(text)

     # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def penal_code():
    load_dotenv()
    pdf_reader = PdfReader("/Users/gabrielrenno/Documents/ASK_PDF/codigo_penal_1ed.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    
    chunks = text_splitter.split_text(text)

     # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF", layout="wide", initial_sidebar_state="expanded")
    st.title("PenalPedia")
    st.sidebar.title("Navigation")

    # Add an about section
    st.sidebar.info(
        "Version 0.0.1 ")
    st.sidebar.write("Developed by: NJTR A.I. Consultancy")
    page = st.sidebar.selectbox("Go to", ["Home", "Ask the Constitution", "Ask the Penal Code", "Ask your PDF"])

    if page == "Home":
        home()
    elif page == "Ask the Constitution":
        ask_the_constitution()
    elif page == "Ask the Penal Code":
        ask_the_penal_code()
    elif page == "Ask your PDF":
        compare_pdf()

def home():
    st.write("PenalPedia - Your Legal Knowledge Companion")
    st.write("""Welcome to PenalPedia, the ultimate app for exploring the Brazilian Constitution and Penal Code. Whether you have questions, need clarification, or seek specific information, PenalPedia is here to assist you on your legal journey.

With a user-friendly interface and a vast database of legal knowledge, PenalPedia empowers you to delve into the intricacies of Brazilian law. The app harnesses advanced technologies to provide accurate and relevant answers to your inquiries about the Constitution and Penal Code.

Key Features:

Ask the Constitution: Wondering about specific constitutional provisions? Ask your questions, and PenalPedia will swiftly guide you through the relevant sections of the Brazilian Constitution.

Ask the Penal Code: Need clarification on criminal law matters? Seek insights into the Brazilian Penal Code through PenalPedia's comprehensive knowledge base. Simply ask your questions, and the app will provide detailed and well-researched responses.

Ask your PDF: Upload your PDF files and ask questions related to their content. PenalPedia will process the uploaded documents, summarize them, and help you generate context-aware questions. Explore legal nuances within the provided context and receive accurate answers.

PenalPedia is backed by the LangChain framework, ensuring a seamless and efficient experience. Our cutting-edge technologies, such as OpenAI embeddings and FAISS vector stores, enable comprehensive document analysis and efficient information retrieval.

Embark on a legal exploration with PenalPedia and unravel the complexities of the Brazilian legal system. Get the answers you seek, gain a deeper understanding of the law, and navigate the intricacies of the Constitution and Penal Code with confidence.

Note: PenalPedia is intended for informational purposes only and should not be considered legal advice. Always consult a qualified legal professional for specific legal matters.

Start your legal journey today with PenalPedia - Your Legal Knowledge Companion!""")

def ask_the_constitution():
    st.write("Ask the Constitution: Wondering about specific constitutional provisions? Ask your questions, and PenalPedia will swiftly guide you through the relevant sections of the Brazilian Constitution.")
    question = st.text_input("Enter your question about the Constitution")
    if st.button("Ask"):
        # Code to process the question and generate response
        response, cb = generate_constitution_response(question)
        st.write("Response:", response)
        st.write("Callback:", cb)

def ask_the_penal_code():
    st.write("Ask the Penal Code: Need clarification on criminal law matters? Seek insights into the Brazilian Penal Code through PenalPedia's comprehensive knowledge base. Simply ask your questions, and the app will provide detailed and well-researched responses.")
    question = st.text_input("Enter your question about the Penal Code")
    if st.button("Ask"):
        # Code to process the question and generate response
        response, cb = generate_penal_code_response(question)
        st.write("Response:", response)
        st.write("Callback:", cb)

def compare_pdf():
    st.write("Ask your PDF: Upload your PDF files and ask questions related to their content. PenalPedia will process the uploaded documents, summarize them, and help you generate context-aware questions. Explore legal nuances within the provided context and receive accurate answers.")
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")
    if uploaded_file is not None:
        # Space for the user to input the question by typing the question
        user_question = st.text_input("Ask a question about your PDF:")
        if st.button("Ask"):
            # Code to process the question and generate response
            response, cb = generate_response_from_pdf_question(uploaded_file, user_question)
            st.write("Response:", response)
            st.write("Callback:", cb)

# Helper functions for processing the questions and generating responses
def generate_constitution_response(question):
    # Code to generate response based on the question about the Constitution
    constituition_knowledge_base = constitution()
    if question:
        docs = constituition_knowledge_base.similarity_search(question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=question)
        return response, cb
   
def generate_penal_code_response(question):
    # Code to generate response based on the question about the Penal 
    penal_code_knowledge_base = penal_code()
    if question:
        docs = penal_code_knowledge_base.similarity_search(question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=question)
        return response, cb


def generate_response_from_pdf_question(uploaded_file, question):
    # Code to process the uploaded PDF file and generate response to the selected question
    penal_code_knowledge_base = penal_code()
    constituition_knowledge_base = constitution()
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()   
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    if question:
        docs = knowledge_base.similarity_search(question)
        docs_constituition = constituition_knowledge_base.similarity_search(question)
        llm = OpenAI()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        pdf_summary = chain.run(input_documents=docs)
        st.write("PDF Summary:", pdf_summary)
        with get_openai_callback() as cb:
          
            template = """
                This is the question about the PDF file:{question}
                The information contained in the PDF file is in: {pdf_summary}.

                Redo the question given that now you know the context of the PDF file, make the context clear in your question.
                """
            prompt = PromptTemplate(
            input_variables=["question", "pdf_summary"],
            template=template)

            final_prompt = prompt.format(question=question, pdf_summary=pdf_summary)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            final_question = chain.run(input_documents=docs, question=final_prompt)

            docs_penal_code = penal_code_knowledge_base.similarity_search(final_question)
            template = """
                Answer this question: {final_question}

                Use this as a context to look for answers: {docs_penal_code}.

                Also use this as a context: {pdf_summary}.
                
                Give a clear answer and explain your answer with sources.

                Do not say anything you are not sure about.

                """
            prompt = PromptTemplate(
            input_variables=["final_question", "docs_penal_code", "pdf_summary"],
            template=template)

            final_prompt = prompt.format(final_question=final_question, docs_penal_code=docs_penal_code, pdf_summary=pdf_summary)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs_penal_code, question=final_prompt)
        return response, cb

if __name__ == "__main__":
    main()
