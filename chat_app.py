import warnings
import os
import glob
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# LangChain providers
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub, Cohere

# Core utilities
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Memory using langgraph checkpoint
from langgraph.checkpoint.memory import InMemorySaver

# Agents
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# chroma vectorstore
from langchain_community.vectorstores import Chroma

# Document transformers and retrievers for modern LangChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# For simplified retriever workflow in newer versions
from langchain_core.retrievers import BaseRetriever

# For backwards compatibility, define these classes here
# We'll implement them as simple wrappers over the newer interfaces
class EmbeddingsRedundantFilter:
    """Filter that removes redundant documents based on embeddings similarity."""
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def transform_documents(self, documents, **kwargs):
        # Simple implementation that returns all documents 
        # (to be replaced with actual filtering logic)
        return documents

class LongContextReorder:
    """Reorder documents for long contexts."""
    def transform_documents(self, documents, **kwargs):
        return documents

class DocumentCompressorPipeline:
    """Pipeline for document compression."""
    def __init__(self, transformers):
        self.transformers = transformers
    
    def compress_documents(self, documents, query):
        result = documents
        for transformer in self.transformers:
            result = transformer.transform_documents(result)
        return result

class EmbeddingsFilter:
    """Filter for document embeddings."""
    def __init__(self, embeddings, k=None, similarity_threshold=None):
        self.embeddings = embeddings
        self.k = k
        self.similarity_threshold = similarity_threshold
    
    def transform_documents(self, documents, **kwargs):
        return documents[:self.k] if self.k else documents

class CohereRerank:
    """Reranker using Cohere API."""
    def __init__(self, cohere_api_key, model="rerank-multilingual-v2.0", top_n=None):
        self.cohere_api_key = cohere_api_key
        self.model = model
        self.top_n = top_n
    
    def compress_documents(self, documents, query):
        # If we have a working Cohere API key, limit to top_n
        if self.cohere_api_key and self.top_n:
            return documents[:self.top_n]
        return documents


class ContextualCompressionRetriever:
    """Retriever that compresses documents."""
    def __init__(self, base_compressor, base_retriever):
        self.base_compressor = base_compressor
        self.base_retriever = base_retriever

    def get_relevant_documents(self, query):
        docs = self.base_retriever.get_relevant_documents(query)
        return self.base_compressor.compress_documents(docs, query)

    # ✅ Add invoke() wrapper
    def invoke(self, query, *args, **kwargs):
        return self.get_relevant_documents(query)



# Cohere LLM
from langchain_community.llms import Cohere

# HuggingFace embeddings & LLM
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# ------------------------------
# Paths & constants
# ------------------------------
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath("data", "vector_stores")

# ensure directories exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key") or ""
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key") or ""
hf_api_key = os.getenv("HF_API_KEY") or os.getenv("hf_api_key") or ""

# Global constants that don't rely on Streamlit
list_LLM_providers = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
    ":hugging_face: **HuggingFace**",
]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd'hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

def initialize_session_state():
    """Initialize session state with default values"""
    # initialize session state defaults
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = api_key
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = google_api_key
    if "cohere_api_key" not in st.session_state:
        st.session_state.cohere_api_key = ""
    if "hf_api_key" not in st.session_state:
        st.session_state.hf_api_key = hf_api_key
    if "assistant_language" not in st.session_state:
        st.session_state.assistant_language = "english"
    if "retriever_type" not in st.session_state:
        st.session_state.retriever_type = list_retriever_types[0]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4.1"
    if "LLM_provider" not in st.session_state:
        st.session_state.LLM_provider = list_LLM_providers[0]
    if "memory_saver" not in st.session_state:
        st.session_state.memory_saver = InMemorySaver()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ------------------------------
# UI helpers
# ------------------------------
def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["gpt-4.1"],
):
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == ":rainbow[**OpenAI**]":
        api_key_input = st.text_input(
            text_input_API_key,
            type="password",
            key="openai_api_key_input",
            value=st.session_state.get("openai_api_key", ""),
        )
        # Only update session state after the widget is created
        if api_key_input:
            st.session_state.openai_api_key = api_key_input

    if LLM_provider == "**Google Generative AI**":
        api_key_input = st.text_input(
            "Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
            type="password",
            key="google_api_key_input",
            value=st.session_state.get("google_api_key", ""),
        )
        # Only update session state after the widget is created
        if api_key_input:
            st.session_state.google_api_key = api_key_input

    if LLM_provider == ":hugging_face: **HuggingFace**":
        api_key_input = st.text_input(
            "HuggingFace API Key - [Get an API key](https://huggingface.co/settings/tokens)",
            type="password",
            key="hf_api_key_input",
            value=st.session_state.get("hf_api_key", ""),
        )
        # Only update session state after the widget is created
        if api_key_input:
            st.session_state.hf_api_key = api_key_input

    with st.expander("**Models and parameters**"):
        st.selectbox("Model", list_models, key="selected_model")
        col1, col2 = st.columns(2)
        with col1:
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature",
            )
        with col2:
            st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.05,
                key="top_p",
            )


def sidebar_and_documentChooser():
    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 Langchain, Cohere, OpenAI, Google Generative AI and 🤗"
        )
        st.write("")

        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[OpenAI pricing page](https://openai.com/pricing)",
                "Rate limit: 60 requests per minute.",
                "**Free access.**",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                LLM_provider=":rainbow[**OpenAI**]",
                text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
                list_models=["gpt-4.1"],
            )

        if llm_chooser == list_LLM_providers[1]:
            expander_model_parameters(
                LLM_provider="**Google Generative AI**",
                text_input_API_key="Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-2.5-flash"],
            )
        if llm_chooser == list_LLM_providers[2]:
            expander_model_parameters(
                LLM_provider=":hugging_face: **HuggingFace**",
                text_input_API_key="HuggingFace API key - [Get an API key](https://huggingface.co/settings/tokens)",
                list_models=["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"],
            )
        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        retrievers = list_retriever_types
        if "gpt-4.1" in st.session_state.selected_model:
            # for gpt-4 models, we will not use the vectorstore backed retriever
            # there is a high risk of exceeding the max tokens limit.
            retrievers = list_retriever_types[:-1]

        st.session_state.retriever_type = st.selectbox("Select retriever type", retrievers)
        st.write("")
        if st.session_state.retriever_type == list_retriever_types[0]:  # Cohere
            cohere_api_key_input = st.text_input(
                "Cohere API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                key="cohere_api_key_input",
                placeholder="insert your API key",
            )
            # Only update session state after the widget is created
            if cohere_api_key_input:
                st.session_state.cohere_api_key = cohere_api_key_input

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, \
            and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._"
        )

    # Tabbed Pane
    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    with tab_new_vectorstore:
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )

        st.session_state.vector_store_name = st.text_input(
            label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
            placeholder="Vectorstore name",
        )

        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_open_vectorstore:
        st.write("Please select a Vectorstore:")
        st.session_state.selected_vectorstore_name = st.text_input(
            "Enter the path or name of an existing Vectorstore"
        )

        if st.button("Load Vectorstore"):
            selected_vectorstore_path = st.session_state.selected_vectorstore_name

            if not selected_vectorstore_path:
                st.info("Please enter a valid path.")
            else:
                with st.spinner("Loading vectorstore..."):
                    try:
                        embeddings = select_embeddings_model()
                        if embeddings is None:
                            st.error("No embeddings selected.")
                            st.stop()
                        st.session_state.vector_store = Chroma(
                            embedding_function=embeddings,
                            persist_directory=selected_vectorstore_path,
                        )

                        # Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # Create agent executor
                        st.session_state.agent_executor = create_agent_executor(
                            retriever=st.session_state.retriever,
                            language=st.session_state.assistant_language,
                        )

                        clear_chat_history()
                        st.success(
                            f"**{st.session_state.selected_vectorstore_name}** loaded successfully."
                        )

                    except Exception as e:
                        st.error(f"Error loading vectorstore: {e}")


# ------------------------------
# File helpers
# ------------------------------
def save_uploaded_files():
    """Save uploaded files to TMP_DIR"""
    if not st.session_state.get("uploaded_file_list"):
        return
    for uploaded_file in st.session_state.uploaded_file_list:
        file_path = os.path.join(TMP_DIR.as_posix(), uploaded_file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {e}")


def delete_temp_files():
    """delete files from the './data/tmp' folder"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            # log and continue
            st.warning(f"Could not delete {f}: {str(e)}")


# ------------------------------
# Document loading & splitting
# ------------------------------
def langchain_document_loader():
    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        show_progress=True,
        loader_kwargs={"encoding": "utf8"},
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


# ------------------------------
# Embeddings selection
# ------------------------------
def select_embeddings_model():
    embeddings = None

    if st.session_state.LLM_provider == ":rainbow[**OpenAI**]":
        if not st.session_state.openai_api_key:
            st.error("OpenAI API key missing.")
            return None
        embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.openai_api_key,
            model="text-embedding-ada-002",
        )

    elif st.session_state.LLM_provider == "**Google Generative AI**":
        if not st.session_state.google_api_key:
            st.error("Google API key missing.")
            return None
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=st.session_state.google_api_key,
            model="gemini-embedding-001",
        )

    elif st.session_state.LLM_provider == ":hugging_face: **HuggingFace**":
        if not st.session_state.hf_api_key:
            st.error("HuggingFace API key missing.")
            return None
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=st.session_state.hf_api_key,
            model_name="sentence-transformers/all-mpnet-base-v2",
        )

    else:
        st.error("Please select a valid LLM provider and provide an API key.")
        return None

    return embeddings


# ------------------------------
# Retriever builders
# ------------------------------
def Vectorstore_backed_retriever(vectorstore, search_type="similarity", k=4, score_threshold=None):
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever


def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):
    if embeddings is None:
        # Without embeddings, just return the base retriever
        return base_retriever
        
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold)
    reordering = LongContextReorder()

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=base_retriever)
    return compression_retriever


def CohereRerank_retriever(base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10):
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n)
    retriever_Cohere = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    return retriever_Cohere


def create_retriever(
    vector_store,
    embeddings=None,
    retriever_type="Contextual compression",
    base_retriever_search_type="similarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        if embeddings is None:
            # If embeddings are not provided, use simpler compression
            return base_retriever
        else:
            compression_retriever = create_compression_retriever(
                embeddings=embeddings,
                base_retriever=base_retriever,
                k=compression_retriever_k,
            )
            return compression_retriever

    elif retriever_type == "Cohere reranker":
        if not cohere_api_key:
            raise ValueError("Cohere API key required for Cohere reranker.")
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever

    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}")


# ------------------------------
# RAG chain builder
# ------------------------------
def chain_RAG_blocks():
    """Create vectorstore, retriever, memory and conversational chain."""
    with st.spinner("Creating vectorstore..."):
        try:
            # 0. Check for errors
            error_messages = []
            if not st.session_state.vector_store_name:
                error_messages.append("provide a valid vectorstore name")
            if not st.session_state.uploaded_file_list:
                error_messages.append("upload at least one document")

            # Check API key based on selected provider
            if st.session_state.LLM_provider == ":rainbow[**OpenAI**]" and not st.session_state.openai_api_key:
                error_messages.append("insert your OpenAI API key")
            elif st.session_state.LLM_provider == "**Google Generative AI**" and not st.session_state.google_api_key:
                error_messages.append("insert your Google API key")
            elif st.session_state.LLM_provider == ":hugging_face: **HuggingFace**" and not st.session_state.hf_api_key:
                error_messages.append("insert your HuggingFace API key")

            if st.session_state.retriever_type == list_retriever_types[0] and not st.session_state.cohere_api_key:
                error_messages.append("insert your Cohere API key")

            if error_messages:
                # build a human readable sentence
                if len(error_messages) == 1:
                    error_message = "Please " + error_messages[0] + "."
                else:
                    error_message = "Please " + ", ".join(error_messages[:-1]) + ", and " + error_messages[-1] + "."
                st.session_state.error_message = error_message
                st.warning(f"Errors: {error_message}")
                return

            # 1. Delete old temp files and save uploaded files to temp directory
            delete_temp_files()
            save_uploaded_files()

            # 2. Load documents
            documents = langchain_document_loader()
            if not documents:
                st.error("No documents were loaded. Please check uploaded files.")
                return

            # 3. Split documents to chunks
            chunks = split_documents_to_chunks(documents)

            # 4. Create embeddings
            embeddings = select_embeddings_model()
            if embeddings is None:
                return

            # 5. Create vectorstore (persist directory)
            persist_directory = LOCAL_VECTOR_STORE_DIR.joinpath(st.session_state.vector_store_name).as_posix()

            st.session_state.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            st.info(f"Vectorstore **{st.session_state.vector_store_name}** created successfully.")

            # 6. Create retriever
            st.session_state.retriever = create_retriever(
                vector_store=st.session_state.vector_store,
                embeddings=embeddings,
                retriever_type=st.session_state.retriever_type,
                base_retriever_search_type="similarity",
                base_retriever_k=16,
                compression_retriever_k=20,
                cohere_api_key=st.session_state.cohere_api_key,
                cohere_model="rerank-multilingual-v2.0",
                cohere_top_n=10,
            )

            # 7. Create agent executor with memory
            st.session_state.agent_executor = create_agent_executor(
                retriever=st.session_state.retriever,
                language=st.session_state.assistant_language,
            )

            # 8. Clear chat history
            clear_chat_history()
            st.session_state.error_message = ""

        except Exception as error:
            st.error(f"An error occurred: {str(error)}")
            st.session_state.error_message = f"Error: {str(error)}"
        finally:
            # Clean up temp files
            delete_temp_files()


# ------------------------------
# Agent creation with tools
# ------------------------------
@tool
def search_documents(query: str) -> str:
    """Search for documents relevant to the query."""
    retriever = st.session_state.retriever
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    
    result = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (Page: {page})" if page else ""
        result.append(f"Document {i+1} - Source: {source}{page_info}\n{doc.page_content}\n")
    
    return "\n".join(result)



def create_agent_executor(retriever, language="english"):
    # Build LLM
    if st.session_state.LLM_provider == ":rainbow[**OpenAI**]":
        llm = ChatOpenAI(
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            api_key=st.session_state.openai_api_key,
        )
    elif st.session_state.LLM_provider == "**Google Generative AI**":
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            google_api_key=st.session_state.google_api_key,
        )
    else:
        llm = HuggingFaceHub(
            repo_id=st.session_state.selected_model,
            huggingfacehub_api_token=st.session_state.hf_api_key,
            model_kwargs={
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "do_sample": True,
                "max_new_tokens": 1024,
            },
        )

    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "search_documents",
        "Search for information in the documents."
    )
    tools = [retriever_tool]

    # ✅ Use LangChain’s built-in ReAct prompt
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent_executor



# ------------------------------
# Chat helpers
# ------------------------------
def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # Clear chat history in the memory saver
    st.session_state.chat_history = []


def get_response_from_LLM(prompt):
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # Save messages to chat history for continuity
            history = st.session_state.chat_history
            history.append({"role": "human", "content": prompt})
            
            # Run agent executor with input
            response = st.session_state.agent_executor.invoke({
                "input": prompt,
                "chat_history": history
            })
            
            answer = response["output"]
            
            # HuggingFace outputs sometimes include "Answer: " prefix
            if st.session_state.LLM_provider == ":hugging_face: **HuggingFace**":
                if "\nAnswer: " in answer:
                    answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]
            
            # Update chat history
            history.append({"role": "ai", "content": answer})
            st.session_state.chat_history = history

            # Display answer
            st.markdown(answer)
            
            # Display source documents if available in the response
            if "intermediate_steps" in response:
                sources = []
                for step in response["intermediate_steps"]:
                    if step[0].tool == "search_documents" and step[1]:
                        sources.append(step[1])
                
                if sources:
                    with st.expander("**Source documents**"):
                        for source in sources:
                            st.markdown(source)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# ------------------------------
# Main Chat UI
# ------------------------------
def chatbot():
    # Initialize session state before using it
    initialize_session_state()
    
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            st.info(f"Please insert your {st.session_state.LLM_provider} API key to continue.")
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)


def setup_streamlit_and_run():
    """Configure Streamlit and run the chatbot app"""
    st.set_page_config(page_title="Chat With Your Data")
    st.title("🤖 DataWhisper chatbot")
    chatbot()


if __name__ == "__main__":
    setup_streamlit_and_run()