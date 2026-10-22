import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CUSTOM CSS --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f6f8fc;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5eaf2;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #4338ca);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-title {
        color: #172554;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    div.stButton > button[kind="primary"] {
        background: #4338ca;
        border: none;
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #3730a3;
    }

    [data-testid="stForm"] {
        background: white;
        padding: 1.25rem;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border: 1px solid #e5eaf2;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- MODEL --------------------

@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_tokens=1200,
    )


# -------------------- PROMPT CHAIN --------------------

PROMPT = PromptTemplate.from_template(
    """
You are PaperLens, an expert academic research assistant.

Summarize the research paper using only the supplied source text
when source text is available.

Paper title: {paper_title}
Summary style: {style}
Target length: approximately {length} words

Requirements:
- Accurately represent the paper's claims and contributions.
- Do not invent experiments, results, statistics, or citations.
- Preserve important technical terminology.
- Explain technical ideas clearly without oversimplifying them.
- Follow the requested summary style.
- If the source text is incomplete, do not imply that you read
  the entire paper.
- If only a title is supplied, clearly indicate that the summary
  is based on general model knowledge rather than supplied text.
- Return the summary only, with useful Markdown formatting.

Source text:
{paper_text}
"""
)

def build_chain():
    return PROMPT | get_llm() | StrOutputParser()


# -------------------- SESSION STATE --------------------

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "summary_metadata" not in st.session_state:
    st.session_state.summary_metadata = {}


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <h1>📚 Research Paper Summarizer</h1>
        <p>
            Turn research papers into clear, structured insights.
            Spend less time reading and more time understanding.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## ⚙️ Summary settings")
    st.caption("Customize your research summary.")

    style = st.selectbox(
        "Summary format",
        [
            "Bullet Points",
            "Paragraph",
            "Key Takeaways",
            "Executive Summary",
        ],
    )

    length = st.slider(
        "Target length (words)",
        min_value=50,
        max_value=500,
        value=150,
        step=25,
    )

    st.divider()

    st.markdown("### 💡 Tips")
    st.caption(
        "For the most accurate results, upload the paper PDF "
        "or provide its text instead of using only its title."
    )

    st.caption("Powered by LangChain + Groq")


# -------------------- PAPER INPUT --------------------

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-title">01 · Select your paper</div>',
        unsafe_allow_html=True,
    )

    paper = st.selectbox(
        "Choose a research paper",
        [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional "
            "Transformers for Language Understanding",
            "GPT-3: Language Models are Few-Shot Learners",
            "Retrieval-Augmented Generation for "
            "Knowledge-Intensive NLP Tasks",
            "Sentence-BERT: Sentence Embeddings using "
            "Siamese BERT-Networks",
            "Other / Custom Paper",
        ],
    )

    if paper == "Other / Custom Paper":
        paper_title = st.text_input(
            "Paper title",
            placeholder="Enter the full research paper title...",
        )
    else:
        paper_title = paper

with right:
    st.markdown(
        '<div class="section-title">02 · Add paper content</div>',
        unsafe_allow_html=True,
    )

    source_type = st.radio(
        "Choose source",
        ["Upload PDF", "Paste text", "Title only"],
        horizontal=True,
    )

    paper_text = ""
    source_ready = True

    if source_type == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload a research paper",
            type=["pdf"],
            help="Text-based PDFs work best.",
        )

        if uploaded_file:
            try:
                reader = PdfReader(uploaded_file)
                paper_text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                ).strip()

                if paper_text:
                    st.success(
                        f"Extracted text from "
                        f"{len(reader.pages)} pages."
                    )
                else:
                    st.warning(
                        "No readable text found. This may be "
                        "a scanned PDF requiring OCR."
                    )
                    source_ready = False

            except Exception as exc:
                st.error(f"Could not read the PDF: {exc}")
                source_ready = False
        else:
            st.info("Upload a PDF to continue.")
            source_ready = False

    elif source_type == "Paste text":
        paper_text = st.text_area(
            "Paper content",
            height=180,
            placeholder="Paste the abstract, introduction, "
            "or full paper text here...",
        )

        if not paper_text.strip():
            st.info("Paste some paper text to continue.")
            source_ready = False

    else:
        st.info(
            "Title-only summaries rely on the model's existing "
            "knowledge and may be less accurate."
        )

# -------------------- GENERATE SUMMARY --------------------

st.divider()

generate = st.button(
    "✨ Generate summary",
    type="primary",
    use_container_width=True,
)

if generate:
    if not paper_title.strip():
        st.error("Please provide a paper title.")

    elif not source_ready:
        st.warning("Please provide a valid paper source.")

    else:
        source = (
            paper_text
            if source_type != "Title only"
            else "[No paper text supplied. Summarize from "
                 "general model knowledge and disclose this.]"
        )

        try:
            with st.spinner(
                "Analyzing the paper and preparing your summary..."
            ):
                chain = build_chain()

                summary = chain.invoke(
                    {
                        "paper_title": paper_title,
                        "style": style,
                        "length": length,
                        "paper_text": source,
                    }
                )

                st.session_state.summary = summary
                st.session_state.summary_metadata = {
                    "title": paper_title,
                    "style": style,
                    "length": length,
                    "source": source_type,
                }

        except Exception as exc:
            st.error(
                "Unable to generate the summary. Check your API "
                "key, model configuration, and internet connection."
            )
            with st.expander("Technical details"):
                st.code(str(exc))


# -------------------- RESULTS --------------------

if st.session_state.summary:
    metadata = st.session_state.summary_metadata

    st.markdown(
        '<div class="section-title">03 · Your summary</div>',
        unsafe_allow_html=True,
    )

    st.subheader(metadata["title"])

    col1, col2, col3 = st.columns(3)

    col1.metric("Format", metadata["style"])
    col2.metric("Target words", metadata["length"])
    col3.metric("Source", metadata["source"])

    st.markdown("")

    with st.container(border=True):
        st.markdown(st.session_state.summary)

    st.download_button(
        "⬇️ Download summary",
        data=st.session_state.summary,
        file_name="paper_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.caption(
        "Review important claims against the original paper "
        "before using the summary in academic work."
    )