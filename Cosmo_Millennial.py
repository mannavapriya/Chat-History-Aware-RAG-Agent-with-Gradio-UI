import subprocess, sys

def install_if_missing(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print(f"🔧 Installing missing package: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for package in ["chromadb", "bs4", "gradio", "langchain", "langchain-community", 
                "langchain-google-genai", "langchain-text-splitters"]:
    install_if_missing(package)

import os
import bs4
import gradio as gr
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY_HERE"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

print("Loading website content ...")
loader = WebBaseLoader(
    web_paths=("https://www.cosmo-millennial.com/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("content"))),
)
docs = loader.load()

if not docs:
    raise ValueError("No documents loaded. Check your WebBaseLoader parsing or site structure.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
if not splits:
    raise ValueError("No text splits created. Check the splitter or input documents.")

print("Creating embeddings and vectorstore ...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are Nomi, a travel assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use five sentences maximum and keep the answer concise.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print("Running initial warm-up query ...")
print(
    conversational_rag_chain.invoke(
        {"input": "What are the advantages of joining Cosmo Millennial App?"},
        config={"configurable": {"session_id": "abc123"}},
    )["answer"]
)

def chat_with_nomi(user_input, session_id="default"):
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response["answer"]

with gr.Blocks() as demo:
    gr.Markdown("# Nomi: Your AI Travel Assistant 🤖")

    session_id_input = gr.Textbox(label="Session ID", value="default", interactive=True)
    chatbox = gr.Chatbot(type="messages")
    user_input = gr.Textbox(label="Your question", placeholder="Ask me about Cosmo Millennial...")
    submit_btn = gr.Button("Send")

    def respond(user_message, session_id, chat_history):
        answer = chat_with_nomi(user_message, session_id)
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})
        return chat_history, ""

    submit_btn.click(respond, inputs=[user_input, session_id_input, chatbox], outputs=[chatbox, user_input])
    user_input.submit(respond, inputs=[user_input, session_id_input, chatbox], outputs=[chatbox, user_input])

demo.launch(share=True, debug=True)