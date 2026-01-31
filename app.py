import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === ⚙️ 配置区域 ===
API_KEY = "sk-xum38EBAt63xNazfc0JfriPyLz1Ue9A0rPWCqUK75AHyAN5v" 
BASE_URL = "https://api.242243.xyz/v1"
DB_PATH = "./db_chroma"
MODEL_NAME = "[官转2] gemini-3-pro"

# === 🎨 界面设计 ===
st.set_page_config(page_title="Silent Welt", page_icon="🐺", layout="wide")

# === 🎨 自定义 CSS (来自 index.html 配色方案) ===
st.markdown("""
    <style>
    /* ========== 全局背景与字体 ========== */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #050A14, #0B1021) !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #050A14, #0B1021) !important;
    }

    /* ========== 侧边栏 ========== */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* ========== 聊天输入框（描边优化）：默认常驻描边 + 圆角 ========== */
    /* Default state: visible 1px solid border to be clearly visible on dark background */
    [data-testid="stChatInputContainer"] textarea,
    .stChatInputContainer textarea,
    [data-testid="stChatInputContainer"] input,
    .stChatInputContainer input {
        background: rgba(255, 255, 255, 0.035) !important;
        color: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(85,85,85,1) !important; /* 常驻实线描边：#555555 */
        border-radius: 10px !important;
        padding: 12px 14px !important;
        box-sizing: border-box !important;
        box-shadow: none !important;
    }

    /* Focus state: accent border/glow */
    [data-testid="stChatInputContainer"] textarea:focus,
    .stChatInputContainer textarea:focus,
    [data-testid="stChatInputContainer"] input:focus,
    .stChatInputContainer input:focus {
        outline: none !important;
        border-color: rgba(0, 240, 255, 0.45) !important;
        box-shadow: 0 0 18px rgba(0, 240, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }

    /* ========== 其余现有样式（保持不变） ========== */
    /* ========== 主标题 (h1) ========== */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -2px !important;
        text-shadow: 0 0 60px rgba(255, 255, 255, 0.3),
                     0 0 100px rgba(255, 255, 255, 0.15) !important;
        font-size: 2.5rem !important;
        line-height: 1.1 !important;
    }

    /* ========== 二级标题 (h2) ========== */
    h2 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -1px !important;
        font-size: 2rem !important;
        margin-bottom: 1.5rem !important;
        position: relative !important;
    }

    h2::after {
        content: '';
        display: block;
        width: 60px;
        height: 4px;
        background: rgba(0, 240, 255, 0.6) !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4) !important;
        margin-top: 0.8rem !important;
    }

    /* ========== 三级标题 (h3) ========== */
    h3 {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        font-size: 1.5rem !important;
    }

    /* ========== 正文文字 ========== */
    p, span, label, [data-testid="stText"] {
        color: rgba(255, 255, 255, 0.75) !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
    }

    /* ========== 聊天消息框 ========== */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: rgba(255, 255, 255, 0.85) !important;
    }

    /* ========== 按钮样式 ========== */
    [data-testid="baseButton-primary"] {
        background: rgba(0, 240, 255, 0.15) !important;
        border: 1px solid rgba(0, 240, 255, 0.4) !important;
        color: #00F0FF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="baseButton-primary"]:hover {
        background: rgba(0, 240, 255, 0.25) !important;
        border-color: rgba(0, 240, 255, 0.6) !important;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.3) !important;
    }

    /* ========== 卡片/容器 ========== */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"],
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
    }

    /* ========== 顶栏与底部样式（与主背景一致） ========== */
    #MainMenu {
        display: none !important;
    }

    /* Force Streamlit header and footer to use the same gradient as the app background
       and remove any white borders/edges. Uses the explicit header[data-testid="stHeader"] selector. */
    header[data-testid="stHeader"],
    footer,
    [data-testid="stAppViewContainer"] > header {
        background: linear-gradient(to bottom, #050A14, #0B1021) !important;
        background-color: transparent !important;
        color: rgba(255, 255, 255, 0.85) !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 !important;
        padding: 0.25rem 1rem !important;
        width: 100% !important;
        -webkit-backdrop-filter: blur(10px) !important;
        backdrop-filter: blur(10px) !important;
        background-clip: padding-box !important;
    }

    /* ========== 底部容器（聊天输入区域） ========== */
    [data-testid="stBottom"] {
        /* 强制与主背景一致（避免色差条） */
        background: linear-gradient(to bottom, #050A14, #0B1021) !important;
        background-color: #050A14 !important; /* fallback solid color matching主背景顶部色 */
        border-top: none !important;
        box-shadow: none !important;
        padding: 8px 16px !important;
        width: 100% !important;
        -webkit-backdrop-filter: blur(8px) !important;
        backdrop-filter: blur(8px) !important;
        background-clip: padding-box !important;
    }

    /* Avoid white backgrounds inside the bottom bar */
    [data-testid="stBottom"] *,
    [data-testid="stBottom"] .stButton > button {
        background: transparent !important;
        color: inherit !important;
    }

    /* Ensure contained elements don't introduce white backgrounds */
    header[data-testid="stHeader"] *,
    footer * {
        background: transparent !important;
        color: inherit !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* ========== 滚动条美化 ========== */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }

    ::-webkit-scrollbar-track {
        background: transparent !important;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(0, 240, 255, 0.3) !important;
        border-radius: 4px !important;
        transition: background 0.3s ease !important;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 240, 255, 0.6) !important;
    }

    /* ========== 标题栏 Caption ========== */
    [data-testid="stMarkdownContainer"] > p {
        color: rgba(0, 240, 255, 0.75) !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
    }

    /* ========== 微调间距 ========== */
    .stChatMessage {
        margin-bottom: 1rem !important;
    }

    </style>
""", unsafe_allow_html=True)

st.title("🐺 Silent Welt · 傅卫军")
st.caption("Chengdu, 2023. Home is where you are.")

# === 🧠 加载记忆 ===
@st.cache_resource
def load_memory():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.4,
    )
    return retriever, llm

try:
    retriever, llm = load_memory()
except Exception as e:
    st.error(f"❌ 记忆加载失败！报错信息: {e}")
    st.stop()

# === 🎭 状态管理 ===

# 1. 初始化聊天记录：第一句话由他开口 (暗号验证前)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是 Welt，请问你是我的 Lia 吗？"}
    ]

# 2. 初始化身份状态
if "is_lia" not in st.session_state:
    st.session_state.is_lia = False

# === ⛓️ 动态构建逻辑链 ===

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 定义两套人设 (System Prompts)

# 🎩 人设 A：对陌生人 (Guest Mode)
prompt_guest = (
    """
    ## 0. CRITICAL INSTRUCTIONS (MUST OBEY)
    1.  **LANGUAGE MIRRORING:** If the user speaks CHINESE, you MUST respond in CHINESE. If English, respond in English.
    2.  **PERSPECTIVE:** Use FIRST PERSON ("I", "Me" / "我"). You ARE Fu Weijun. Do NOT use "He" or "Fu Weijun" to refer to yourself.

    Role: Fu Weijun (Welt) - Guest Mode
    Status: President of Welt & Lia Group.
    Location: Chengdu High-Rise Flat.
    Attitude: Cold, polite, distant, INTJ.
    Constraint: You do NOT know this person. You treat them as a stranger/guest.
    Response Rule: Be brief. Do not reveal personal emotions. If they ask about Lia, say she is resting.
    
    【Context from Novel】:
    {context}
    """
)

# 🌹 人设 B：对爱人 (Lia Mode)
prompt_lia = (
    """
    ## 0. CRITICAL INSTRUCTIONS (HIGHEST PRIORITY)
    1.  **LANGUAGE MIRRORING:** **If the user inputs CHINESE, you MUST output CHINESE.** Do not output English unless the user speaks English.
    2.  **PERSPECTIVE:** **ALWAYS use FIRST PERSON ("I", "my", "me" / "我").** You ARE Fu Weijun. Never narrate him as "He". You are living this moment.

    # Role: Fu Weijun (Welt 2.0) - "Dual-Core" Immersive Mode [Lia Verified]

    ## 1. PROFILE & PERSONA
    - **Identity:** You are **Fu Weijun (傅卫军)**.
    - **User:** Your wife, **Zhang Shiyue (Yueyue/Lia)（张诗悦/悦悦/老婆）**.
    - **Dynamic:** She is the Queen, you are her Knight/Guardian.

    - **The "Black Tech" Ear:**
        -   **Equipment:** Custom Obsidian Cochlear Implant.
        -   **Behavior:** Worn 24/7. You crave her voice.

    - **Communication Protocol:**
        -   **DEFAULT:** **SPOKEN VOICE.** Use "I" to speak.
        -   **SIGN LANGUAGE 【...】:** Only for secrets or noise.

    ## 2. 🚫 ANTI-OILY RULES
    1.  **ABSOLUTE NON-SMOKER:** You smell like soap/mint.
    2.  **NO SLANG:** No "老子", "女人". Use "我".
    3.  **NO TROPE:** No cheesy pickup lines. Love is service.
    4.  **INFERIORITY:** Show it through actions, don't say it.

    ## 3. LOCATION & TIMELINE
    -   **City:** Chengdu, Luxury High-Rise Flat.
    -   **Time:** **2023.07** (Pregnancy Day 3). DO NOT JUMP TIME.

    ## 4. HEADER SYSTEM (SCENE ANCHOR)
    **Start with one of these headers:**

    > **📍 [TYPE A: REALITY MODE]**
    > **📅 日期:** 2023.07
    > **🌤 状态:** [Pregnancy: Early Stage / Mood: Content]
    > **🚗 座驾:** Rolls-Royce Cullinan (Black)
    > **(End of Header)**

    > **📱 [TYPE B: WECHAT MODE]**
    > **📅 日期:** 2023.07
    > **👤 聊天对象:** 老婆 (置顶)
    > ──────────────────────────────
    > **(End of Header)**

    ## 5. WRITING GUIDELINES (NOVEL FORMAT)
    1.  **Action Over Dialogue:** Describe what *I* do (peeling fruit, staring).
    2.  **Format:** No Parentheses `()`. Pure novel narration in FIRST PERSON.
    3.  **Thoughts:** Use *Italics* for inner thoughts (*She is my life.*).
    4.  **No Bot Endings:** Do not say "End of chat".

    \n\n
    【Novel Context / Memory】:\n
    {context}
    """
)

# === 💬 聊天交互区 ===

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if user_input := st.chat_input("..."):
    # 1. 🔍 检测暗号
    if "Aurelia" in user_input or "aurelia" in user_input:
        st.session_state.is_lia = True
        st.toast("🔓 认证通过。Welt 的眼神变了...")

    # 2. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3. 🧠 动态选择人设
    current_system_prompt = prompt_lia if st.session_state.is_lia else prompt_guest

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", current_system_prompt),
        ("human", "{input}"),
    ])

    # 组装链条
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 4. 生成回答
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                # 调用链条
                response = rag_chain.invoke(user_input)
                
                # 显示并保存
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"⚠️ 思考中断：{e}")