import gradio as gr
from app.ingest import ingest_document
from app.inference import generate_answer
import json
from datetime import datetime
import os


def main():
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
                label="📂 Session Name (Load or Save)",
                choices=list_sessions(),
                interactive=True,
                elem_id="session-dropdown",
                allow_custom_value=True
            )
            refresh_btn = gr.Button("🔄 Refresh Session List")
            save_btn = gr.Button("💾 Save Session")

        chatbot = gr.Chatbot(
            elem_id="chat-window",
            show_copy_button=True,
            bubble_full_width=False,
            render_markdown=True
        )
        textbox = gr.Textbox(
            label="💬 Ask a Question",
            placeholder="Type your query here and press Enter...",
            interactive=True
        )

        clear_btn = gr.Button("🧹 Clear Chat")
        export_md_btn = gr.Button("📥 Export as Markdown")

        def ask_and_store(query, history, session):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history = history or []
            answer = generate_answer(query)
            history.append((f"{query}\n🕒 {timestamp}", answer))
            if session:
                save_chat_history(session, history)
            return history, gr.update(value="")

        def handle_upload(files, session):
            results = []
            history = []
            for file in files:
                try:
                    ingest_document(file.name)
                    results.append(f"✅ {os.path.basename(file.name)}")
                    history.append((f"Ingested: {os.path.basename(file.name)}", "✅ Success"))
                except Exception as e:
                    results.append(f"❌ {os.path.basename(file.name)} - {str(e)}")
                    history.append((f"Ingested: {os.path.basename(file.name)}", f"❌ Failed: {str(e)}"))
            if session:
                save_chat_history(session, history)
            return history

        def load_session(session_name):
            path = os.path.join("sessions", f"{session_name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
            return []

        def save_session_manual(history, session_name):
            print("Saving session: %s", session_name)
            if session_name == "New Chat":
                session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            save_chat_history(session_name, history)
            return gr.update(choices=list_sessions())

        file_upload.change(fn=handle_upload, inputs=[file_upload, session_dropdown], outputs=chatbot)
        session_dropdown.change(fn=load_session, inputs=session_dropdown, outputs=chatbot)
        textbox.submit(fn=ask_and_store, inputs=[textbox, chatbot, session_dropdown], outputs=[chatbot, textbox])
        export_md_btn.click(fn=lambda h: export_chat(h, format="markdown"), inputs=chatbot)
        clear_btn.click(lambda: [], outputs=chatbot)
        refresh_btn.click(fn=lambda: gr.update(choices=list_sessions()), outputs=session_dropdown)
        save_btn.click(fn=save_session_manual, inputs=[chatbot, session_dropdown], outputs=[session_dropdown])

    return demo


def list_sessions(directory="sessions"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    sessions = sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))
    print("Sessions: %s", sessions)
    if 'New Chat' not in sessions:
        sessions.insert(0, 'New Chat')
    return sessions


def save_chat_history(session_name, history, directory="sessions"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, f"{session_name}.json"), "w") as f:
        json.dump(history, f, indent=2)


def export_chat(history, format="markdown", export_dir="exports"):
    import uuid

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    file_id = uuid.uuid4().hex
    filename = os.path.join(export_dir, f"chat_{file_id}.{format}")

    with open(filename, "w") as f:
        for q, a in history:
            f.write(f"**Q:** {q}\n\n**A:** {a}\n\n")

    return filename


if __name__ == "__main__":
    main().launch()
