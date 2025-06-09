"""
Main entry point for launching the Gradio application.

This script is used when running ragify-docs directly from source.
"""

import logging
from pathlib import Path
import os
import json
from datetime import datetime
import gradio as gr

from ragify_docs.config import settings
from ragify_docs.ingest import ingest_document
from ragify_docs.inference import generate_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(settings.log_dir) / 'ragify_docs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def list_sessions(directory=settings.session_dir):
    """List all available chat sessions from the session directory.

    Args:
        directory (str, optional): Path to the directory containing session files.
            Defaults to settings.session_dir.

    Returns:
        list: Sorted list of session names with 'New Chat' as the first item.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created session directory: {directory}")

        sessions = sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))
        if 'New Chat' not in sessions:
            sessions.insert(0, 'New Chat')
        logger.debug(f"Found {len(sessions)} sessions in {directory}")
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions in {directory}: {str(e)}")
        return ['New Chat']


def save_chat_history(session_name, history, directory=settings.session_dir):
    """Save chat history to a JSON file in the specified directory.

    Args:
        session_name (str): Name of the session to save.
        history (list): List of message dicts with 'role' and 'content' keys.
        directory (str, optional): Directory to save the session file.
            Defaults to settings.session_dir.

    Returns:
        bool: True if save was successful, False otherwise.
    """
    try:
        if not session_name or not history:
            logger.warning("No session name or history provided")
            return False

        # Validate message format
        for msg in history:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                logger.error(f"Invalid message format in history: {msg}")
                return False

        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{session_name}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved session: {session_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving session {session_name}: {str(e)}", exc_info=True)
        return False


def export_chat(history, format="markdown", export_dir=settings.export_dir):
    """Export chat history to a markdown file.

    Args:
        history (list): List of message tuples (user_msg, bot_msg) to export.
        format (str, optional): Export format. Currently only 'markdown' is supported.
            Defaults to "markdown".
        export_dir (str, optional): Directory to save the exported file.
            Defaults to settings.export_dir.

    Returns:
        str: Success message with the path to the exported file.
    """
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(export_dir, f"chat_export_{timestamp}.{format}")

    with open(filepath, "w") as f:
        if format == "markdown":
            f.write("# Chat Export\n\n")
            for i, (user_msg, bot_msg) in enumerate(history, 1):
                f.write(f"## Message {i}\n")
                f.write(f"**User**: {user_msg}\n\n")
                f.write(f"**Assistant**: {bot_msg}\n\n")

    return f"✅ Chat exported to {filepath}"


def main():
    """Create and configure the Gradio interface for the RAGify Docs application.

    Returns:
        gr.Blocks: Configured Gradio Blocks interface.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 RAGify Docs: Domain-Aware Chatbot")

        with gr.Row():
            file_upload = gr.File(
                label="📄 Upload PDF/DOCX/Google Doc",
                file_types=[".pdf", ".docx"],
                file_count="multiple",
                interactive=True,
                elem_id="upload-btn",
                show_label=True
            )

        with gr.Row():
            session_dropdown = gr.Dropdown(
                label="📂 Session Name (Load from saved sessions)",
                choices=list_sessions(),
                interactive=True,
                elem_id="session-dropdown",
                allow_custom_value=True
            )
            refresh_btn = gr.Button("🔄 Refresh Session List")
            session_name = gr.Textbox(
                label="Session Name",
                placeholder="Enter a name for this session",
                scale=2,
                min_width=200
            )
            save_btn = gr.Button("💾 Save Session")

        chatbot = gr.Chatbot(
            label="Chat",
            height=600,
            show_copy_button=True,
            show_share_button=True,
            type="messages"
        )
        textbox = gr.Textbox(
            label="💬 Ask a Question",
            placeholder="Type your query here and press Enter...",
            interactive=True
        )

        clear_btn = gr.Button("🧹 Clear Chat")
        export_md_btn = gr.Button("📥 Export as Markdown")

        def ask_and_store(query, history, session):
            """Process a user query, generate a response, and update chat history.

            Args:
                query (str): User's input query.
                history (list): Current chat history as list of message dicts.
                session (str): Current session name.

            Returns:
                tuple: Updated history and reset textbox value.
            """
            try:
                logger.info(f"Processing query: {query[:100]}...")
                history = history or []

                # Add user message to history
                history.append({"role": "user", "content": query})

                # Generate response
                response = generate_answer(query)

                # Add assistant response to history
                history.append({"role": "assistant", "content": response})

                # Save to session if applicable
                if session and session != 'New Chat':
                    save_chat_history(session, history)

                return history, gr.update(value="")

            except Exception as e:
                logger.critical(f"Unexpected error in ask_and_store: {str(e)}", exc_info=True)
                error_msg = "An unexpected error occurred. Please try again."
                return (history or []) + [(query, error_msg)], gr.update(value="")

        def handle_upload(files, session):
            """Handle file uploads and process documents.

            Args:
                files (list): List of uploaded files.
                session (str): Current session name.

            Returns:
                tuple: Updated history and session dropdown state.
            """
            if not files:
                return [], gr.update()
            results = []
            history = []
            for file in files:
                try:
                    ingest_document(file.name)
                    results.append(f"✅ {os.path.basename(file.name)}")
                    history.append({"role": "user", "content": f"📄 Uploaded: {os.path.basename(file.name)}"})
                except Exception as e:
                    error_msg = f"Error processing {os.path.basename(file.name)}: {str(e)}"
                    logger.error(error_msg)
                    results.append(f"❌ {os.path.basename(file.name)}: {str(e)}")
                    history.append({"role": "assistant", "content": error_msg})

            # If we have a session name, save the history
            if session and session.strip():
                # Auto-generate a session name if 'New Chat' is selected
                if session == 'New Chat':
                    session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    logger.info(f"Auto-generated session name: {session}")

                if save_chat_history(session, history):
                    # Return both the history and the new session name to update the UI
                    return history, session

            return history, gr.update()

        def load_session(session_name):
            """Load chat history for a given session name.

            Args:
                session_name (str): Name of the session to load.

            Returns:
                list: List of message dicts with 'role' and 'content' keys.
            """
            if session_name == "New Chat":
                return []

            path = os.path.join(settings.session_dir, f"{session_name}.json")
            if not os.path.exists(path):
                logger.warning(f"Session file not found: {path}")
                return []

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # Convert old tuple format to new message format if needed
                    if history and isinstance(history[0], (list, tuple)):
                        new_history = []
                        for user_msg, bot_msg in history:
                            if user_msg:
                                new_history.append({"role": "user", "content": user_msg})
                            if bot_msg:
                                new_history.append({"role": "assistant", "content": bot_msg})
                        history = new_history
                logger.info(f"Loaded session: {session_name} with {len(history)} messages")
                return history
            except Exception as e:
                logger.error(f"Error loading session {session_name}: {str(e)}")
                return []

        def save_session_manual(history, session_name):
            """Manually save the current chat session.

            Args:
                history (list): Current chat history as list of message dicts.
                session_name (str): Name to save the session as.

            Returns:
                dict: Updated session dropdown choices.
            """
            try:
                logger.info(f"Attempting to save session: {session_name}")
                if not session_name or not session_name.strip():
                    logger.warning("Empty session name provided")
                    return gr.update(choices=list_sessions())

                if session_name == "New Chat":
                    session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    logger.info(f"Auto-generated session name: {session_name}")

                if save_chat_history(session_name, history):
                    logger.info(f"Successfully saved session: {session_name}")
                else:
                    logger.error(f"Failed to save session: {session_name}")

                return gr.update(choices=list_sessions())

            except Exception as e:
                logger.error(f"Error in save_session_manual: {str(e)}", exc_info=True)
                return gr.update(choices=list_sessions())

        def export_to_markdown(history, session_name):
            """Export chat history to a markdown file.

            Args:
                history (list): Chat history as list of message dicts.
                session_name (str): Name of the session to export.

            Returns:
                str: Path to the exported file or error message.
            """
            if not history:
                return "No history to export"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{session_name}_{timestamp}.md"
            filepath = os.path.join(settings.export_dir, filename)

            try:
                os.makedirs(settings.export_dir, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Chat Export: {session_name}\n\n")
                    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("---\n\n")

                    for msg in history:
                        role = msg.get('role', '').capitalize()
                        content = msg.get('content', '')
                        if role and content:
                            f.write(f"## {role}\n{content}\n\n")
                            f.write("---\n\n")

                logger.info(f"Exported chat history to {filepath}")
                return f"Exported to {filename}"
            except Exception as e:
                error_msg = f"Error exporting chat: {str(e)}"
                logger.error(error_msg)
                return error_msg

        file_upload.change(
            fn=handle_upload,
            inputs=[file_upload, session_dropdown],
            outputs=[chatbot, session_dropdown],
            show_progress=True,
            queue=True
        )
        session_dropdown.change(
            fn=load_session,
            inputs=session_dropdown,
            outputs=chatbot
        )
        textbox.submit(
            fn=ask_and_store,
            inputs=[textbox, chatbot, session_dropdown],
            outputs=[chatbot, textbox]
        )
        export_md_btn.click(
            fn=lambda h: export_chat(h, format="markdown"),
            inputs=chatbot
        )
        clear_btn.click(lambda: [], outputs=chatbot)
        refresh_btn.click(
            fn=lambda: gr.update(choices=list_sessions()),
            outputs=session_dropdown
        )
        save_btn.click(
            fn=save_session_manual,
            inputs=[chatbot, session_name],
            outputs=session_dropdown
        )

    return demo


def launch():
    """Launch the Gradio web interface.

    This is the main entry point for running the application.
    """
    main().launch()
