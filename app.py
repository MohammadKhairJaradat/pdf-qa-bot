"""Local Gradio interface for the PDF QA bot."""

import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from qa_bot import GEMINI_MODE, LOCAL_MODE, InputError, retriever_qa


def ask_pdf(file, question, mode):
    try:
        return retriever_qa(file, question, mode)
    except InputError as error:
        return f"Input error: {error}", ""
    except Exception:
        # Provider errors may contain account or request details.
        return (
            "The request could not be completed. For Gemini mode, check your "
            "API key, model access, quota, and connection. "
            "If the service is busy, try again later.",
            "",
        )


sample = str(Path(__file__).parent / "sample_data" / "reporting_handbook.pdf")

demo = gr.Interface(
    fn=ask_pdf,
    inputs=[
        gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Question", placeholder="What is the reporting deadline?"),
        gr.Dropdown([LOCAL_MODE, GEMINI_MODE], value=LOCAL_MODE, label="Mode"),
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=5),
        gr.Textbox(label="Retrieved passages", lines=14),
    ],
    title="PDF Question Answering Bot",
    description=(
        "Find information in a PDF and check its source pages. "
        "Local mode runs without an API key and shows matching passages. "
        "Gemini mode generates answers and sends extracted text "
        "and your question to Google. Text-based PDFs only; up to 10 MB and 100 pages."
    ),
    examples=[
        [sample, "What is the monthly reporting deadline?", LOCAL_MODE],
        [sample, "How is net revenue calculated?", LOCAL_MODE],
        [sample, "Who reviews the dashboard before publication?", LOCAL_MODE],
    ],
    cache_examples=False,
    flagging_mode="never",
    api_name="ask_pdf",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
