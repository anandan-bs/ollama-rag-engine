"""
Edu PDF Chatbot Demo

This demo shows how to use the RAGChatbot class to create a Gradio interface
for a chatbot that answers questions based on the text from uploaded PDFs or
text files.
"""

import os
import json
import logging
import gradio as gr
from datetime import datetime
import shutil

from app.rag_ollama import RAGChatbot
from app.health import get_system_health
from app.config import STORAGE_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

rag = RAGChatbot(STORAGE_CONFIG["upload_dir"])

def check_health():
    """Check system health and return status message."""
    try:
        health = get_system_health()
        all_healthy = all(component["status"] == "healthy" for component in health.values())
        if all_healthy:
            return "✅ All systems operational"
        else:
            unhealthy = [name for name, status in health.items() if status["status"] != "healthy"]
            return f"⚠️ Issues detected with: {', '.join(unhealthy)}"
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return f"❌ Health check failed: {str(e)}"

def chat_fn(message, history):
    """
    Process a message from the user and return a response.

    Args:
        message (str): The message from the user.
        history (list): The chat history.

    Returns:
        list: The updated chat history.
        str: The response from the chatbot.
        list: The updated chat history.
    """
    if not message:
        return history, "", history
    answer = rag.query(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return history, "", history


def save_convo(history, chat_name=""):
    logger.info(f"Saving conversation with name: {chat_name}")
    """
    Save the chat history to a file.

    Args:
        history (list): The chat history.
        chat_name (str): The name of the chat (optional).

    Returns:
        str: The name of the saved file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if chat_name:
        safe_name = ''.join(c for c in chat_name.replace(
            ' ', '_') if c.isalnum() or c in '_-')
        filename = f"{safe_name}_{timestamp}.json"
    else:
        filename = f"chat_{timestamp}.json"
    path = os.path.join(STORAGE_CONFIG["chatlog_dir"], filename)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    display_name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {os.path.splitext(filename)[0]}"
    return display_name


def get_chat_list():
    logger.debug("Fetching chat list")
    """
    Get a list of saved chat files.

    Returns:
        list: A list of tuples containing the display name and filename.
    """
    files = sorted(
        [f for f in os.listdir(STORAGE_CONFIG["chatlog_dir"]) if f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(STORAGE_CONFIG["chatlog_dir"], x)),
        reverse=True
    )
    chats = [("New Chat", None)]
    for filename in files:
        path = os.path.join(STORAGE_CONFIG["chatlog_dir"], filename)
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        name = os.path.splitext(filename)[0]
        display = f"{mtime.strftime('%Y-%m-%d %H:%M:%S')} - {name}"
        chats.append((display, filename))
    return chats


def update_chat_dropdown():
    """
    Update the chat dropdown list.

    Returns:
        list: The list of display names.
    """
    chat_list = get_chat_list()
    choices = [chat[0] for chat in chat_list]
    return choices


def on_save_click(history, chat_name):
    """
    Save the chat history when the save button is clicked.

    Args:
        history (list): The chat history.
        chat_name (str): The name of the chat (optional).

    Returns:
        list: The updated list of display names.
        str: The name of the saved file.
    """
    new_display = save_convo(history, chat_name)
    choices = update_chat_dropdown()
    return gr.update(choices=choices, value=new_display)


def load_chat(display_name):
    logger.info(f"Loading chat: {display_name}")
    """
    Load a saved chat history.

    Args:
        display_name (str): The name of the chat history.

    Returns:
        list: The loaded chat history.
        list: The loaded chat history.
    """
    if display_name == "New Chat" or not display_name:
        return [], []

    chat_list = get_chat_list()
    filename = next((f for d, f in chat_list if d == display_name), None)

    if filename and os.path.exists(os.path.join(STORAGE_CONFIG["chatlog_dir"], filename)):
        with open(os.path.join(STORAGE_CONFIG["chatlog_dir"], filename)) as f:
            raw_data = json.load(f)

        cleaned = []
        for msg in raw_data:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                cleaned.append(
                    {"role": msg["role"], "content": msg["content"]})
            elif isinstance(msg, list) and len(msg) == 2:
                cleaned.append({"role": "user", "content": msg[0]})
                cleaned.append({"role": "assistant", "content": msg[1]})
            else:
                print(f"Skipping malformed entry: {msg}")
        return cleaned, cleaned

    return [], []


def index_documents(files):
    logger.info(f"Indexing {len(files) if files else 0} documents")
    """
    Index the uploaded PDFs or text files.

    Args:
        files (list): The list of uploaded files.

    Returns:
        str: The result of the upload operation.
    """
    if not files:
        return "⚠️ No files selected."
    for file in files:
        shutil.move(file.name, os.path.join(STORAGE_CONFIG["upload_dir"], os.path.basename(file.name)))
    try:
        rag.build_index()
        return f"✅ Uploaded {len(files)} file(s)."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Check system health before starting
health_status = check_health()
logger.info(f"Initial health check: {health_status}")

with gr.Blocks() as demo:
    state = gr.State([])

    # Upload domain specific documents
    with gr.Accordion("📂 Load Domain Documents", open=False):
        upload_files = gr.File(file_types=[
                               ".pdf", ".txt"], file_count="multiple", label="Upload PDFs or Text Files")
        upload_button = gr.Button("📄 Upload")
        upload_output = gr.Textbox(label="Upload Result", lines=2)
        upload_button.click(index_documents, inputs=[
                            upload_files], outputs=[upload_output])

    with gr.Row():
        chatbox = gr.Chatbot(label="📡 RAG Chatbot",
                             value=[], type="messages", height=400)
        chat_sessions = gr.Dropdown(
            choices=[c[0] for c in get_chat_list()],
            value="New Chat",
            label="📜 Chat History",
            interactive=True,
            scale=1
        )

    msg = gr.Textbox(placeholder="Ask anything...")

    with gr.Row():
        submit_btn = gr.Button("Send")
        save_btn = gr.Button("💾 Save Chat")
        name_input = gr.Textbox(
            placeholder="Session name (optional)", label="Chat Name")

    # Chat logic
    submit_btn.click(chat_fn, [msg, state], [state, msg, chatbox])
    save_btn.click(on_save_click, inputs=[
                   state, name_input], outputs=[chat_sessions])
    chat_sessions.change(fn=load_chat, inputs=chat_sessions,
                         outputs=[chatbox, state])

demo.launch()
