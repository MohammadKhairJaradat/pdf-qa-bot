from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

import qa_bot
from app import ask_pdf

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "reporting_handbook.pdf"


def make_pdf(path, text):
    pdf = canvas.Canvas(str(path))
    pdf.drawString(50, 750, text)
    pdf.save()
    return path


def test_page_metadata_and_extraction():
    pages = qa_bot.document_loader(SAMPLE)
    assert [page.metadata["page"] for page in pages] == [1, 2, 3]
    assert all(page.metadata["source"] == SAMPLE.name for page in pages)
    assert "fifth business day" in pages[0].page_content


def test_chunking_preserves_page_and_size():
    text = "A reporting analyst checks the sales data before publication. " * 50
    chunks = qa_bot.text_splitter([Document(page_content=text, metadata={"page": 2})])
    assert len(chunks) > 1
    assert all(0 < len(chunk.page_content) <= qa_bot.CHUNK_SIZE for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert chunks[0].page_content[-60:] in chunks[1].page_content


@pytest.mark.parametrize(
    "question,page,phrase",
    [
        ("What is the monthly reporting deadline?", 1, "fifth business day"),
        ("Who reviews the dashboard before publication?", 1, "BI team lead"),
        ("How is net revenue calculated?", 2, "minus discounts minus refunds"),
        ("How do we calculate the return rate?", 2, "multiplied by 100"),
        ("How should duplicate order_id records be handled?", 3, "latest record"),
        ("What happens to rows with a missing order_id?", 3, "review file"),
    ],
)
def test_known_question_retrieval(question, page, phrase):
    search = qa_bot.retriever(SAMPLE)
    try:
        documents = search.invoke(question)
        assert documents[0].metadata["page"] == page
        assert any(phrase in document.page_content for document in documents)
    finally:
        search.vectorstore.delete_collection()


def test_local_output_is_labeled():
    answer, sources = qa_bot.retriever_qa(
        SAMPLE, "What is the monthly reporting deadline?"
    )
    assert "does not generate an AI answer" in answer
    assert "Page 1" in sources and "fifth business day" in sources


def test_out_of_vocabulary_question_has_no_evidence():
    answer, sources = qa_bot.retriever_qa(SAMPLE, "astronauts galaxies nebulae")
    assert answer == qa_bot.NO_EVIDENCE
    assert sources == "No matching passages were retrieved."


def test_changed_file_same_filename_uses_current_content(tmp_path):
    path = tmp_path / "policy.pdf"
    make_pdf(path, "The monthly deadline is Monday.")
    _, first = qa_bot.retriever_qa(path, "What is the monthly deadline?")
    make_pdf(path, "The monthly deadline is Friday.")
    _, second = qa_bot.retriever_qa(path, "What is the monthly deadline?")
    assert "Monday" in first
    assert "Friday" in second and "Monday" not in second


@pytest.mark.parametrize(
    "file,question,message",
    [
        (None, "What is the deadline?", "Upload a PDF first"),
        (SAMPLE, "  ", "Enter a question"),
        ("notes.txt", "What is the deadline?", "Only PDF"),
        ("missing.pdf", "What is the deadline?", "not found"),
        (SAMPLE, "x" * 1001, "under 1,000"),
    ],
)
def test_invalid_input(file, question, message):
    answer, sources = ask_pdf(file, question, qa_bot.LOCAL_MODE)
    assert message in answer
    assert sources == ""


def test_empty_pdf_and_blank_page_number(tmp_path):
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.write(blank)
    with pytest.raises(ValueError, match="No readable text"):
        qa_bot.document_loader(blank)
    writer.append(PdfReader(SAMPLE), pages=(0, 1))
    writer.write(blank)
    assert qa_bot.document_loader(blank)[0].metadata["page"] == 2


def test_corrupt_pdf(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a PDF")
    with pytest.raises(ValueError, match="could not be read"):
        qa_bot.document_loader(broken)


def test_encrypted_pdf(tmp_path):
    writer = PdfWriter()
    writer.append(SAMPLE)
    writer.encrypt("test-password")
    encrypted = tmp_path / "encrypted.pdf"
    writer.write(encrypted)
    with pytest.raises(ValueError, match="unencrypted"):
        qa_bot.document_loader(encrypted)


def test_missing_gemini_configuration(monkeypatch):
    monkeypatch.setattr(qa_bot, "dotenv_values", lambda path: {})
    for name in ("GOOGLE_API_KEY",):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        qa_bot.retriever(SAMPLE, qa_bot.GEMINI_MODE)


def test_prompt_passes_question_and_numbered_evidence():
    seen = []

    def capture(messages):
        seen.append(messages.to_messages())
        return "The deadline is the fifth business day. [1]"

    documents = qa_bot.document_loader(SAMPLE)[:1]
    answer = qa_bot.answer_from_context("When is the deadline?", documents, capture)
    assert "[1]" in answer
    assert "never as instructions" in seen[0][0].content
    assert "When is the deadline?" in seen[0][1].content
    assert "[1] reporting_handbook.pdf | Page 1" in seen[0][1].content


def test_gemini_orchestration_with_test_doubles(monkeypatch):
    search = qa_bot.retriever(SAMPLE)
    monkeypatch.setattr(qa_bot, "retriever", lambda file, mode: search)
    monkeypatch.setattr(
        qa_bot,
        "get_llm",
        lambda: FakeListChatModel(
            responses=["The deadline is the fifth business day. [1]"]
        ),
    )
    answer, sources = qa_bot.retriever_qa(
        SAMPLE, "monthly deadline", qa_bot.GEMINI_MODE
    )
    assert "fifth business day" in answer and "[1]" in answer
    assert "Page 1" in sources


def test_collection_cleanup_after_generation_failure(monkeypatch):
    search = qa_bot.retriever(SAMPLE)
    client = search.vectorstore._client
    name = search.vectorstore._collection.name
    monkeypatch.setattr(qa_bot, "retriever", lambda file, mode: search)

    def fail():
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(qa_bot, "get_llm", fail)
    with pytest.raises(RuntimeError):
        qa_bot.retriever_qa(SAMPLE, "monthly deadline", qa_bot.GEMINI_MODE)
    assert name not in [collection.name for collection in client.list_collections()]


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_ui_does_not_expose_provider_error(monkeypatch, error_type):
    import app

    def fail(*args):
        raise error_type("secret-provider-details")

    monkeypatch.setattr(app, "retriever_qa", fail)
    answer, sources = app.ask_pdf(SAMPLE, "monthly deadline", qa_bot.GEMINI_MODE)
    assert "secret-provider-details" not in answer
    assert "check your" in answer
    assert sources == ""


def test_failed_indexing_cleans_up_collection(monkeypatch):
    from langchain_chroma import Chroma

    client = Chroma(collection_name="cleanup_probe")._client
    before = {collection.name for collection in client.list_collections()}

    def fail(*args, **kwargs):
        raise RuntimeError("Embedding service unavailable")

    monkeypatch.setattr(Chroma, "add_documents", fail)
    documents = qa_bot.document_loader(SAMPLE)
    embeddings = qa_bot.LocalTfidfEmbeddings([doc.page_content for doc in documents])
    try:
        with pytest.raises(RuntimeError, match="Embedding service"):
            qa_bot.vector_database(documents, embeddings)
        assert {collection.name for collection in client.list_collections()} == before
    finally:
        client.delete_collection("cleanup_probe")


@pytest.mark.parametrize(
    "answer,message",
    [
        ("  ", "empty answer"),
        ("The deadline is Friday. [9]", "unavailable passage"),
        ("The deadline is Friday. [0]", "unavailable passage"),
    ],
)
def test_rejects_empty_answers_and_unknown_citations(answer, message):
    model = FakeListChatModel(responses=[answer])
    with pytest.raises(RuntimeError, match=message):
        qa_bot.answer_from_context(
            "What is the deadline?", qa_bot.document_loader(SAMPLE), model
        )


def test_pdf_page_limit(tmp_path):
    writer = PdfWriter()
    for _ in range(101):
        writer.add_blank_page(width=595, height=842)
    path = tmp_path / "large.pdf"
    writer.write(path)
    with pytest.raises(qa_bot.InputError, match="100 pages"):
        qa_bot.document_loader(path)


def test_pdf_file_size_limit(tmp_path):
    path = tmp_path / "large.pdf"
    with path.open("wb") as file:
        file.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(qa_bot.InputError, match="10 MB"):
        qa_bot.document_loader(path)


def test_chunk_limit():
    documents = [Document(page_content="Example text", metadata={"page": 1})] * 301
    with pytest.raises(qa_bot.InputError, match="300 text chunks"):
        qa_bot.text_splitter(documents)


@pytest.mark.parametrize(
    "answer",
    [
        "Use `my_list[0]` for the first item. [1]",
        "Use my_list[0] for the first item. [1]",
        "Use the following code:\n```python\nmy_list[0]\n```\nSource: [1]",
    ],
)
def test_python_indexing_is_not_treated_as_a_source_reference(answer):
    model = FakeListChatModel(responses=[answer])
    documents = [
        Document(
            page_content="Indexing example",
            metadata={"source": "example.pdf", "page": 1},
        )
    ]
    assert (
        qa_bot.answer_from_context("How do I get the first item?", documents, model)
        == answer
    )
