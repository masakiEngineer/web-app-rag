import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain  # ←ここを変更
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --- APIキーの読み込み ---
def load_api_key() -> str:
    """
    .env ファイルから OpenAI API キーを読み込む

    Returns:
        str: APIキー
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("`.env` に `OPENAI_API_KEY` を設定してください。")
    return api_key

# --- ドキュメントの読み込みと分割 ---
def load_and_split_documents(uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    アップロードされたファイルを読み込み、チャンクに分割

    Args:
        uploaded_file: Streamlit のアップロードファイルオブジェクト
        chunk_size (int): 各チャンクのサイズ
        chunk_overlap (int): チャンク間のオーバーラップ

    Returns:
        list: 分割された Document オブジェクトのリスト
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = TextLoader(tmp_path, encoding='utf-8')
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# --- ベクトルストアとConversationalRetrievalChainの作成 ---
def build_qa_chain(docs: list, api_key: str, k: int = 3):
    """
    ドキュメントからベクトルDBとConversationalRetrievalChainを構築

    Args:
        docs (list): Document オブジェクトのリスト
        api_key (str): OpenAI APIキー
        k (int): 検索対象とする上位文書数

    Returns:
        ConversationalRetrievalChain: 会話形式QAチェーン
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_db = FAISS.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(openai_api_key=api_key)
    return ConversationalRetrievalChain.from_llm(llm, retriever)

# --- チャットUIの表示と処理 ---
def run_chat_ui(qa_chain):
    """
    ユーザーとのチャットUI処理を行う

    Args:
        qa_chain: ConversationalRetrievalChainインスタンス
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 会話履歴を画面に表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("💬 質問を入力してください"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            pairs = []
            messages = st.session_state.chat_history
            length = len(messages)
            if length % 2 != 0:
                length -= 1
            for i in range(0, length, 2):
                pairs.append((messages[i]["content"], messages[i+1]["content"]))

            result = qa_chain({"question": prompt, "chat_history": pairs})
            response = result["answer"]
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

        with st.expander("📚 使用された文書"):
            for idx, doc in enumerate(result.get("source_documents", [])):
                st.markdown(f"**{idx+1}.** {doc.page_content[:300]}...")

# --- メイン処理 ---
def main():
    api_key = load_api_key()
    st.set_page_config(page_title="RAGチャットアプリ", layout="wide")

    st.markdown("""
        <style>
            body { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); font-family: 'Segoe UI', sans-serif; }
            .main { background-color: rgba(255, 255, 255, 0.85); border-radius: 20px; padding: 30px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2); }
            h1 { text-align: center; color: #ff5e62; font-size: 48px; margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1> RAG チャットアプリ</h1>", unsafe_allow_html=True)
    st.markdown('<div class="main">', unsafe_allow_html=True)

    if api_key:
        uploaded_file = st.file_uploader("🔽 検索対象のファイルをアップロードしてください", type=["txt", "md", "pdf"])
        if uploaded_file:
            docs = load_and_split_documents(uploaded_file)
            qa_chain = build_qa_chain(docs, api_key)
            run_chat_ui(qa_chain)
        else:
            st.info("🔍 まずは検索対象のファイルをアップロードしてください。")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()