import os
# === 🔧 修复点：使用了新的导入路径 ===
from langchain_community.document_loaders import TextLoader
# 旧写法：from langchain.text_splitter import RecursiveCharacterTextSplitter
# 新写法 (你装的版本需要这个)：
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# === 配置区域 ===
# 你的小说放在 data 文件夹，数据库会生成在 db_chroma 文件夹
DATA_PATH = "./data"
DB_PATH = "./db_chroma"

def create_vector_db():
    print("🚀 [1/4] 系统启动：开始扫描小说文件...")
    
    # 1. 读取文件
    documents = []
    # 再次确认路径，防止找不到
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到 {DATA_PATH} 文件夹！")
        return

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"📖 发现文件: {file}")
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ 无法读取 {file}: {e}")

    if not documents:
        print("❌ 并没有在 data 文件夹里找到 txt 文件！")
        return
    
    print(f"✅ 成功读取！共 {len(documents)} 个文件。")

    # 2. 切分文本
    print("✂️ [2/4] 正在进行精细切分 (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"✅ 切分完成！你的小说被切成了 {len(texts)} 个记忆碎片。")

    # 3. 向量化
    print("🧠 [3/4] 正在下载模型并转化记忆 (第一次运行这步最慢，请耐心等待)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. 存入数据库
    print("💾 [4/4] 正在写入 Chroma 数据库...")
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
        
    vector_db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("-" * 30)
    print("🎉 大功告成！AI 已经记住了你的小说！")

if __name__ == "__main__":
    create_vector_db()