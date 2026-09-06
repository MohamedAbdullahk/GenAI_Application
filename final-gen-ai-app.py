import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 LangChain Chat Assistant")

# 2. Load API Key
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OPENAI_API_KEY. Please check your .env file.")
    st.stop()

# 3. Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Existing Chat Messages on Re-render
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Helper Function to Build the Chain
@st.cache_resource
def get_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and friendly AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return prompt | llm | StrOutputParser()

chain = get_chain()

# 6. Handle New User Input
if user_input := st.chat_input("Type your message here..."):
    # Render user message on screen immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Store user input in message history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare historical context for LangChain (converting to role tuples)
    history_tuples = [
        (msg["role"], msg["content"]) 
        for msg in st.session_state.messages[:-1] # Exclude the prompt just added
    ]

    # Render streaming assistant response
    with st.chat_message("assistant"):
        # st.write_stream streams generator outputs directly to the UI
        response_stream = chain.stream({
            "history": history_tuples,
            "input": user_input
        })
        full_response = st.write_stream(response_stream)

    # Store assistant response in message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})