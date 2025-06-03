import pytest
import app
import json
import logging
import os

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag"))
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "auto_test.log")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

test_questions = [
    "Hi, introduce Carro.",
    "What services does Carro provide?",
    "How does Carro calculate the price of a used car?",
    "What did I eat for lunch?",
    "Tell me more about Carro's financing options.",
    "How can I use the Carro platform to buy a car?",
]

@pytest.mark.parametrize("question", test_questions)
def test_chatbot(question):
    history = []
    response = ""
    for chunk in app.chat_fn(question, history):
        response = chunk
        
    print(f"\nQ: {question}\nMemory: {history}\nA: {response}\n{'-'*80}\n")
    logging.info(json.dumps({
        "question": question,
        "history": history,
        "response": response,
    }, ensure_ascii=False))
    
    logging.shutdown()
    assert len(response) > 0, "Bot response should not be empty!"
    

