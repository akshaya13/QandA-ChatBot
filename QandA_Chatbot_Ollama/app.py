import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import Ollama


import os
from dotenv import load_dotenv

load_dotenv()
## Langsmith tracking

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):    
    llm = Ollama(model = llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title
st.title("Enhanced Q&A Chatbot with Ollama")

## Sidebar for API-KEY
st.sidebar.title("Settings")

## Dropdown to select the preferred model
llm = st.sidebar.selectbox("Select an Ollama Model", 
                           ("llama3", "llama2", "gemma:2b")
                           )

model_info = {
    "llama3": "Llama3: Suitable for general queries and conversations.",
    "llama2": "Llama2: Good balance between performance and accuracy.",
    "gemma:2b": "Gemma:2b: Optimized for technical and complex questions."
}
st.sidebar.write(model_info[llm])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value = 0.7)

max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value =150)


## Main Interface for User input

st.write("Ask me Anything!")
user_input = st.text_input("You: ")
submit_button = st.button("Submit")


if submit_button:
    if user_input:
        with st.spinner('Generating response...'):
            response = generate_response(user_input, llm, temperature, max_tokens)
        st.write(response)
    else:
        st.warning("Please provide the query !")
