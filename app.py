# app.py
import streamlit as st
from dotenv import load_dotenv
import pickle
                           
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
# get callback to get stats on query cost
from langchain.callbacks import get_openai_callback
import os
load_dotenv() 
# Sidebar contents
with st.sidebar:
    st.title('💬 PDF Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered PDF chatbot built using:
    - [Streamlit](https://streamlit.io/) Frontend Framework
    - [LangChain](https://python.langchain.com/) App Framework
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [FAISS](https://github.com/facebookresearch/faiss) vector store
 
    ''')

    st.markdown('## Environment Variables')
    openai_api_key = st.text_input("Enter OPENAI_API_KEY")
    if openai_api_key:  # Check if the input is not empty
        os.environ["OPENAI_API_KEY"] = openai_api_key  # Set the environment variable

# Main function
def main():
    
    st.title("Embedding Loader and Question Answering")
    
    # Initialize session state to store conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
                 

    # Allow users to ask questions
    question = st.text_input("Ask a question:")
    if st.button("Submit"):
        if question:
            # Process user question and generate response
            response = process_question(question)
            # Append question and response to conversation history
            st.session_state.conversation.append({'question': question, 'response': response})
                                                        
                                           
        else:
            st.warning("Please enter a question.")
    
    # Display conversation history
    st.subheader("Conversation History")
    for entry in st.session_state.conversation:
        st.write(f"**User:** {entry['question']}")
        st.write(f"**Bot:** {entry['response']}")
        st.write("---")

# Function to generate response to user's question
def process_question(question):
    embeddings = get_embeddings()
    if embeddings is None:
        return "Error: Embeddings not loaded."
    
    # Retrieve relevant documents
    docs = retrieve_docs(question, embeddings)
    
    # Generate response using LangChain
    response = generate_response(docs, question)
    
    return response

# Function to retrieve relevant documents
def retrieve_docs(question, embeddings):
    docs = embeddings.similarity_search(question, k=3)
    if len(docs) == 0:
        raise Exception("No documents found")
    else:
        return docs

# Function to generate response
def generate_response(docs, question):
    llm = ChatOpenAI(temperature=0.0, max_tokens=1000, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)
    return response

                                                          
                                           
                                                                         
                                                                       
                                                                    
                                                                            

# Function to get embeddings
def get_embeddings():
    # Update with your path to embeddings file
                                                 
                                                     
                                                       
                                                          
                              
                             

    embeddings_path = "s.pkl"  
    
    if os.path.exists(embeddings_path):
        embeddings = load_embeddings(embeddings_path)
                          
        return embeddings
    else:
        st.error("Embeddings file not found!")
                                 
        return None

# Function to load embeddings from a pickle file
def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings
    
if __name__ == "__main__":
    main()
