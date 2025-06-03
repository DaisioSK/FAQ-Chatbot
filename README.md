
# Carro FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for Carro, the online used car dealership, to enhance customer experience.  
The chatbot answers user questions based on an official FAQ document and, for real-time queries, integrates internet search APIs to provide up-to-date information.

---

## 📝 Project Overview

This chatbot combines document-based question answering and real-time internet search, powered by modern LLMs (e.g., Ollama, OpenAI) and the LangChain framework.  
Users can interact through a web-based Gradio interface. The bot provides accurate, clear, and polite responses, referencing the source of answers when possible.

---

## ✨ Key Features & Highlights

- **FAQ-First QA**: Uses Carro's provided FAQ PDF as the primary knowledge source for in-scope queries.
- **Internet Search Integration**: For questions outside the FAQ (e.g., real-time pricing), calls Google Custom Search API and displays the latest online results.
- **Retrieval-Augmented Generation (RAG)**: Hybrid approach enhances both answer accuracy and coverage.
- **Streaming LLM Responses**: Powered by Ollama (Llama3) or OpenAI (GPT-4o-mini); easily switchable.
- **Robust Error Handling**: User-friendly, informative fallback messages for unclear queries, network/API failures, or missing data.
- **Session Memory**: Summarizes recent conversations to maintain context across turns.
- **Simple Web UI**: Fast and user-friendly Gradio chat interface; ready for demonstration or further customization.
- **Automated Testing**: `pytest`-based script with multi-turn scenario coverage and auto-logging.

---

## 🧩 Challenges & Solutions

Building a production-ready RAG chatbot for FAQ and real-time queries involves a range of technical and product challenges. Below summarizes key pain points encountered in this project, and the concrete solutions designed to address them.

### 1. **Challenge: Reliable FAQ Extraction from Noisy PDF**
- **Problem:** Original FAQ documents may contain noisy formatting, watermarks, page numbers, and inconsistent section headers, all of which negatively impact downstream retrieval quality.
- **Solution:**  
  - Developed custom text cleaning and structure-aware chunking pipeline (`doc_ingest.py`), including regular expressions to remove noise, and rules to extract section titles.
  - Applied hierarchical chunking: FAQ is segmented by semantic structure (Level 1/2/3 headers), then merged and split for both context richness and granularity.

### 2. **Challenge: Accurate Retrieval in Multi-Turn Dialogue**
- **Problem:** Simple vector search may return irrelevant or incomplete chunks, especially when queries are vague or span multiple FAQ topics.
- **Solution:**  
  - Utilized a strong sentence embedding model (`all-MiniLM-L6-v2`) for dense retrieval.
  - Embedded chunk metadata (section titles) into both chunk content and index, so LLM and user see context.
  - Maintained a session memory summary (last 6 rounds) to provide continuity across user turns.

### 3. **Challenge: Real-Time Data Requests Not Covered by FAQ**
- **Problem:** Some user questions (e.g., “What is the current interest rate?”) require up-to-date info that FAQ can’t provide.
- **Solution:**  
  - Integrated Google Custom Search API and SerpAPI for fallback to real-time web results.
  - Automatic context blending: If no relevant FAQ chunk is found, bot gracefully switches to online results, still citing the source and maintaining a friendly tone.

### 4. **Challenge: Robust and Polite Error Handling**
- **Problem:** Network, API, or LLM errors can create user frustration or expose technical jargon.
- **Solution:**  
  - Comprehensive try/except handling at every I/O and API call stage.
  - User-facing error messages are always polite, informative, and suggest next steps (e.g., try again, contact support).


### 5. **Challenge: Easy Setup and Repeatable Testing**
- **Problem:** Ensuring fast onboarding for reviewers and repeatable, realistic testing of the chatbot’s capabilities.
- **Solution:**  
  - Centralized environment config using `.env` for easy API key and path management.
  - Automated test script (`test_chat.py`) covers a wide range of typical and edge-case queries, with result logging for traceability.

### 6. **Challenge: Keeping Answers Trustworthy and Transparent**
- **Problem:** Users may want to know where answers come from, especially for sensitive or policy-related questions.
- **Solution:**  
  - The bot always references the relevant FAQ section or states “according to online search,” providing both the answer and its provenance.

> These strategies help the chatbot meet the assignment's goals for accuracy, completeness, resilience, and user experience, making it suitable for real-world deployment or further expansion.

---

## 🚀 Quickstart: Setup & Installation

### 1. Clone the Repo & Prepare Data

```bash
git clone <your-repo-url>
cd <project-folder>
# Place the FAQ PDF into the designated data folder (see .env settings)
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
# (Recommended: Python 3.9+)
```

### 3. Environment Variables

Create a `.env` file in your root directory with the following keys (see template below):

```env
DATA_DIR=./data              # Path to your PDF FAQ files
RAG_DIR=./rag                # Where logs/results will be stored
INDEX_DIR=./faiss_index      # Where the FAISS vectorstore will be saved
RAG_DIST_THRESHOLD=0.8       # Retrieval threshold (tune as needed)
GOOGLE_API_KEY=xxx           # Google Custom Search API key
GOOGLE_CSE_ID=xxx            # Google CSE ID
SERP_API_KEY=xxx             # (optional) SerpAPI key
OPENAI_API_KEY=xxx           # (optional) For GPT-4o-mini, else use Ollama by default
```

You can just edit the  `.env-example` provided and rename it to `.env` at your own workspace. 

### 4. Build the Vector Store

RAG will automatically create index file of the FAQ documents during the first run. 
The full Carro Malaysia Terms of Use can be viewed at: 
https://carro.co/my/en/terms

---

### 5. Launch the Chatbot (Gradio UI)

```bash
cd src
python app.py
# The Gradio interface will be available at http://localhost:7860
```

---

## 🗂️ Project Structure

```
├── src/
├──── app.py               # Main entrypoint; Gradio webchat + controller logic
├──── api_online_search.py # Google/SerpAPI integration for real-time search
├──── doc_ingest.py        # PDF reading, chunking, cleaning, embedding, indexing
├──── doc_retrieve.py      # Vectorstore loading and RAG retrieval logic
├──── test_chat.py         # Automated test cases & logging for chatbot validation
├── docs/                # Place your FAQ PDF(s) here
├── rag/                 # Log files and conversation records
├──── faiss_index/         # Generated vector index
├── requirements.txt     # Python package dependencies
└── README.md            # (this file)
```

---

## 🧪 Testing Methodology

- **Automated Test Script**:  
  Run `pytest -s test_chat.py` for batch Q&A testing, logging, and result validation.
- **Scenario Coverage**:  
  Test cases cover:
    - FAQ-answerable questions
    - Out-of-scope (e.g., unrelated or ambiguous) questions
    - Real-time queries (pricing, availability)
    - Error and fallback scenarios

---

## 💡 Approach & Implementation Notes

- **Document QA**:  
  - PDFs are cleaned and split into semantic chunks with section titles for improved retrieval granularity.
  - FAISS vectorstore enables fast, scalable dense retrieval.
- **Hybrid Retrieval**:  
  - If document retrieval fails or confidence is low, triggers a fallback to online search.
- **LLM Reasoning**:  
  - Llama3 via Ollama is used by default (configurable).  
  - The model is instructed to answer strictly based on provided context and always reference source sections.
  - OpenAI LLM is also available. Please uncomment specific lines of code and add your API key in the `.env` file
- **User Experience**:  
  - Responses avoid jargon and are friendly; error messages are graceful and informative.
- **Session Memory**:  
  - Summaries of the last 6 turns help maintain context in multi-turn dialogues.

---

## 🧩 Setup Tips & Troubleshooting

- **Missing Data**: Ensure your FAQ PDF is present and environment variables are correct.
- **API Keys**: Google Custom Search/SerpAPI keys are required for internet search fallback.
- **Index Rebuilding**: Delete `faiss_index/` if you want to force a fresh ingest.
- **Port Conflicts**: Default Gradio port is `7860`. Change via code if needed.

---

## 🏆 Matching Assignment Requirements

- **Document-based QA** ✔️  
- **Internet Search for Live Data** ✔️  
- **Polite, Contextual Responses** ✔️  
- **Robust Error Handling** ✔️  
- **Simple, Usable Interface** ✔️  
- **Extensive Testing & Logging** ✔️  
- **Clean, Well-Commented Code** ✔️  
- **Easy Setup & Cross-Platform** ✔️

---

## 🙋 FAQ

**Q: What if my question isn't in the FAQ?**  
A: The bot will attempt an online search for up-to-date answers, or politely indicate if it's unable to assist.

**Q: How are sources indicated?**  
A: For FAQ answers, the chatbot cites relevant section titles or content for easy reference.

**Q: Can I switch the LLM?**  
A: Yes, easily switch between Ollama (Llama3) and OpenAI (GPT-4o-mini) in `app.py`.

**Q: How do I add more FAQs?**  
A: Place new PDFs into your `data/` directory and re-run ingestion.

---

## ⚙️ Further Improvements

- Add few-shot examples to make the prompting more robust. 
- Add chunk deduplication and cross-encoder reranking for even higher retrieval quality.
- Use more advanced PDF parsing tools (such as Unstructured) to capture more accurate section hierarchy information from PDF documents
- More sophisticated UI (streamlit/webapp) if desired.
- Dockerization for one-click deployment.

---

## 📋 License & Disclaimer

All content, code, and documentation provided in this project are the intellectual property of their respective creators and are intended solely for evaluation and educational purposes.

Unauthorized reproduction, distribution, or use for commercial purposes is strictly prohibited without prior written consent.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [OpenAI](https://platform.openai.com/)
- [Gradio](https://www.gradio.app/)
