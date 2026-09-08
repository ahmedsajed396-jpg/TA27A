
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(
    page_title="A2A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background: var(--background-color);
    color: var(--text-color);
}

.block-container {
    max-width: 1200px;
    padding-top: 70px;
    padding-bottom: 50px;
}

section[data-testid="stSidebar"] {
    background: var(--secondary-background-color);
}

section[data-testid="stSidebar"] * {
    color: var(--text-color);
}

.logo {
    width: 58px;
    height: 58px;
    border-radius: 17px;
    background: linear-gradient(
        135deg,
        #6366f1,
        #06b6d4
    );
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 15px;
    box-shadow:
        0 10px 30px rgba(99, 102, 241, 0.25);
}

.a2a-title {
    color: var(--text-color);
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
    letter-spacing: -1px;
}

.a2a-subtitle {
    color: var(--text-color);
    opacity: 0.65;
    font-size: 15px;
}

.online {
    display: inline-block;
    margin-top: 15px;
    padding: 7px 13px;
    border-radius: 20px;
    background: rgba(34, 197, 94, 0.12);
    color: #16a34a;
    font-size: 13px;
}

.card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.20);
    border-radius: 18px;
    padding: 22px;
    margin-top: 18px;
    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.05);
}

.card-title {
    color: var(--text-color);
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-text {
    color: var(--text-color);
    opacity: 0.65;
    font-size: 13px;
    line-height: 1.6;
}

.file-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.20);
    border-radius: 16px;
    padding: 16px;
    margin-top: 20px;
}

.file-name {
    color: var(--text-color);
    font-weight: 700;
}

.file-info {
    color: var(--text-color);
    opacity: 0.60;
    font-size: 12px;
    margin-top: 4px;
}

div[data-testid="stChatMessage"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 16px;
    margin-bottom: 12px;
}

div[data-testid="stChatInput"] {
    margin-top: 15px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    background: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
}

.stButton > button:hover {
    border-color: #6366f1;
}

.agent-box {
    background: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 12px;
    padding: 11px;
    margin-bottom: 8px;
}

.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.45;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(128, 128, 128, 0.20);
}

</style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# PDF EXTRACTION
# =========================

def extract_pdf(file):
    reader = PdfReader(file)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    full_text = "\n\n".join(pages)

    return full_text, len(reader.pages)


# =========================
# OPENAI / DEEPSEEK CLIENT
# =========================

def get_client():

    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]

    except Exception:
        return None

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


# =========================
# AI
# =========================

def ask_ai(document, question, instructions):

    client = get_client()

    if client is None:
        return None, "API_KEY"

    document = document[:80000]

    system_message = """
You are A2A, an advanced AI document intelligence assistant.

Your job is to analyze uploaded documents and answer user questions accurately.

Use the uploaded document as the primary source.

If the requested information is not available in the document,
clearly tell the user that it was not found.

Be professional, accurate and easy to understand.

Use headings and bullet points when useful.
"""

    user_message = f"""
Additional instructions:

{instructions}

Uploaded document:

{document}

User question:

{question}
"""

    try:

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        return answer, None

    except Exception as error:

        return None, str(error)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown(
        '<div class="logo">🤖</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-size:25px;font-weight:800;color:var(--text-color);">A2A</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:var(--text-color);opacity:0.55;font-size:12px;">Agent to Agent Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        if uploaded_file.name != st.session_state.document_name:

            try:

                text, pages = extract_pdf(uploaded_file)

                if text.strip():

                    st.session_state.document_text = text
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.page_count = pages
                    st.session_state.messages = []

                    st.success("PDF ready")

                else:

                    st.error(
                        "No readable text was found in this PDF."
                    )

            except Exception as error:

                st.error(
                    "Could not read the PDF."
                )

                st.write(str(error))

    st.divider()

    # =========================
    # A2A AGENTS
    # =========================

    st.markdown("### 🧠 A2A Agents")

    st.markdown("📄 **Document Agent**")
    st.caption("Ready")

    st.markdown("🔍 **Analysis Agent**")
    st.caption("Ready")

    st.markdown("💬 **Answer Agent**")
    st.caption("Ready")

    st.divider()

    # =========================
    # AI INSTRUCTIONS
    # =========================

    st.markdown("### 📝 AI Instructions")

    custom_prompt = st.text_area(
        "Instructions",
        placeholder="Example: Summarize this document...",
        height=120,
        label_visibility="collapsed"
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================
# MAIN HEADER
# =========================

st.markdown(
    '<div class="logo">🤖</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="a2a-title">A2A</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="a2a-subtitle">Agent to Agent · AI Document Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="online">● AI System Online</div>',
    unsafe_allow_html=True
)


# =========================
# HOME SCREEN
# =========================

if not st.session_state.document_text:

    st.markdown("")

    st.markdown("## 📄 Welcome to A2A")

    st.write(
        "Upload a PDF document and start an intelligent "
        "conversation with your document."
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📄")

        st.markdown("**Document Analysis**")

        st.caption(
            "Read and analyze information from your PDF."
        )

    with col2:

        st.markdown("### 🔍")

        st.markdown("**Smart Search**")

        st.caption(
            "Ask questions using natural language."
        )

    with col3:

        st.markdown("### 🤖")

        st.markdown("**AI Agents**")

        st.caption(
            "A2A is designed for multiple AI agents working together."
        )


# =========================
# DOCUMENT CHAT
# =========================

else:

    st.markdown(
        f"""
        <div class="file-card">

            <div style="font-size:25px;">
                📕
            </div>

            <div style="margin-top:7px;">

                <div class="file-name">
                    {st.session_state.document_name}
                </div>

                <div class="file-info">
                    PDF · {st.session_state.page_count} pages · Ready
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    st.markdown(
        """
        <div style="
            color:var(--text-color);
            font-size:24px;
            font-weight:700;
            margin-top:25px;
        ">
            💬 Chat with your document
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:var(--text-color);
            opacity:0.55;
            font-size:13px;
            margin-top:5px;
            margin-bottom:20px;
        ">
            Ask A2A anything about your uploaded document.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # CHAT HISTORY
    # =========================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # =========================
    # CHAT INPUT
    # =========================

    question = st.chat_input(
        "Ask A2A about your document..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "A2A is analyzing your document..."
            ):

                answer, error = ask_ai(
                    st.session_state.document_text,
                    question,
                    custom_prompt
                )

                if error == "API_KEY":

                    st.error(
                        "DEEPSEEK_API_KEY was not found in Streamlit Secrets."
                    )

                elif error:

                    st.error(
                        "AI connection error:"
                    )

                    st.write(error)

                else:

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


# =========================
# FOOTER
# =========================

st.markdown(
    """
    <div class="footer">
        A2A · Agent to Agent Intelligence
    </div>
    """,
    unsafe_allow_html=True
)