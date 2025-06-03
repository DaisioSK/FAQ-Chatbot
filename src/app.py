import os
import threading
import gradio as gr
import json
import socket
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from api_online_search import google_custom_search
from doc_retrieve import retrieve_documents
from doc_ingest import ingest_documents
import logging


# memory class - for storing conversation summary
class SummaryMemory:
    def __init__(self):
        self.summary = ""
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.summary

    def set(self, value):
        with self.lock:
            self.summary = value
            


def get_env_variable(var_name):
    value = os.environ.get(var_name)
    if not value:
        raise EnvironmentError(f"Environment variable {var_name} is not set.")
    return value


def get_llm():
    return Ollama(
        model="llama3",
        temperature=0.1,
        top_k=40,
        top_p=0.9,
        num_predict=512
    )
    
    
def get_openai_llm(api_key):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.0,
        streaming=True,
    )


def log_response(query, history, context, response, context_from_rag):
    logging.info(json.dumps({
        "query": query,
        "history": history,
        "context_snippet": context[:120],
        "retrieved_titles": context_from_rag,
        "response": response,
    }, ensure_ascii=False))
    
    
# async update summary function
def update_summary(history):
    llm = get_llm()
    # up to 6 rounds of conversation history
    conv_history = ""
    for user, ai in history[-6:]:
        conv_history += f"User: {user}\nAssistant: {ai}\n"
    summary_prompt = (
        "Summarize the following conversation history in 2-3 sentences. "
        "Retain all important facts, user intents, and context for future reference:\n"
        f"{conv_history}"
    )
    messages = [
        {"role": "system", "content": "You are a conversation summarizer."},
        {"role": "user", "content": summary_prompt}
    ]
    # call LLM to generate summary
    summary = llm.invoke(messages)
    summary_memory.set(summary)


def chat_fn(query, history):
    
    context = ""
    context_from_rag = ""
    
    try:
        
        llm = get_llm()
        # llm = get_openai_llm(OPENAI_API_KEY)

        # RAG search from documents
        rag_results = retrieve_documents(INDEX_DIR, query, 10, {"rag_dist_threshold": float(RAG_DIST_THRESHOLD)})
        context_from_rag = f"\n{'-'*80}\n".join([doc.page_content for doc in rag_results]) if rag_results else ""
        
        if not context_from_rag:
            context = "There is no relevant FAQ information found. "
            api_saerch_results = google_custom_search(query, GOOGLE_API_KEY, GOOGLE_CSE_ID)
            context_from_api = "\n".join([f"[{item['title']}]\n{item['snippet']}\n" for item in api_saerch_results if 'snippet' in item]) if api_saerch_results else ""
            context += f"Here are some online search results:\n{context_from_api}" if context_from_api else "Online search API is facing some problems as well. Please kindly tell user to try again later or contact customer support for assistance."
        else:
            context = f"Please answer my question based on the following information retrieved from FAQ documents only. If you cannot find the answer, please say you don't know politely." + context_from_rag

        # fetch the latest summary
        summary = summary_memory.get()
        
        system_prompt = (
            "You are a helpful assistant named 'Corol'. Only answer based on the provided context information."
            "Once you answer according to FAQ documents, you need to indicate which section or content you got the information from, so that user can refer to the original document."
            "If information is insufficient, please say you are not sure and ask for more details.\n"
            f"[Summary] Here is the previous conversation summary (if any):\n{summary}\n"
            f"[Context] Here is the context information you are asked to refer to solely:\n{context}"
        )
        user_prompt = f"{query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print(messages)

        bot_response = ""
    
        # Ollama streaming
        for chunk in llm.stream(messages):
            bot_response += chunk
            yield bot_response
        
        # # OpenAI streaming
        # for chunk in llm.stream(messages):
        #     if hasattr(chunk, "content") and chunk.content:
        #         bot_response += chunk.content
        #         yield bot_response
        
    except ValueError as e:
        bot_response = "It seems your question is not clear or is outside the supported scope. Please clarify or rephrase."
        logging.warning(f"ValueError: {str(e)}", exc_info=True)
        yield bot_response
    except (requests.exceptions.Timeout, socket.timeout) as e:
        bot_response = "Sorry, our system is experiencing a temporary network timeout. Please try again in a moment."
        logging.error(f"TimeoutError: {str(e)}", exc_info=True)
        yield bot_response
    except (requests.exceptions.ConnectionError, socket.gaierror) as e:
        bot_response = "Sorry, our chatbot could not connect to the backend. Please try again later."
        logging.error(f"ConnectionError: {str(e)}", exc_info=True)
        yield bot_response
    except RuntimeError as e:
        bot_response = "Sorry, the system encountered an internal error. Please try again or contact support."
        logging.error(f"RuntimeError: {str(e)}", exc_info=True)
        yield bot_response
    except Exception as e:
        # 兜底处理未知异常
        bot_response = (
            "Sorry, something went wrong. "
            "Our team has been notified. Please try again or contact customer service."
        )
        logging.error(f"UnhandledException: {str(e)}", exc_info=True)
        yield bot_response
        

    # update history with user query and bot response
    def update_summary_async(history):
        try:
            update_summary(history)
        except Exception as e:
            print(f"Failed to update summary: {e}")
                
    log_response(query, history, context, bot_response, context_from_rag)
    threading.Thread(target=update_summary_async, args=(history + [(query, bot_response)],)).start()


# load environment variables
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / get_env_variable("DATA_DIR")
RAG_DIR = BASE_DIR / get_env_variable("RAG_DIR")
INDEX_DIR = BASE_DIR / get_env_variable("INDEX_DIR")
RAG_DIST_THRESHOLD = get_env_variable("RAG_DIST_THRESHOLD")
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY")
GOOGLE_CSE_ID = get_env_variable("GOOGLE_CSE_ID")
SERP_API_KEY = get_env_variable("SERP_API_KEY")
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")



if __name__ == "__main__":

    summary_memory = SummaryMemory()
    logging.basicConfig(
        filename=f"{RAG_DIR}/rag_test.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # read and load documents into vectorstore
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist. Please check your environment variables.")
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index directory {INDEX_DIR} does not exist. Please check your environment variables.")
    ingest_documents(DATA_DIR, INDEX_DIR)

    # Gradio chat interface
    chatbot = gr.ChatInterface(
        fn=chat_fn,
        title="Carro FAQ Chatbot - Corol",
        description="Support Streaming & Multi-window memories to provide information according to Q&A Documents and Online Search Results."
    )
    chatbot.launch()
