import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import  OpenAIEmbeddings,ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


DEEPSEEK_API_KEY = "sk-"
PDF_PATH = "your_document.pdf"

VECTOR_DB_PATH = "D:./vector"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_LLM_MODEL = "deepseek-chat"
DEEPSEEK_EMBEDDING_MODEL = "text-embedding-ada-002"

def chunking(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n",".","!","?","，", "。", "！", "？", " "]
    )
    split_chunks = text_splitter.split_documents(documents)

    print(f"PDF 解析完成,共切分为{len(chunks)}个文本片段")
    return split_chunks

def create_vectors(split_chunks,use_deepseek_embedding = True):
    if use_deepseek_embedding:
        embeddings = OpenAIEmbeddings(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model_name=DEEPSEEK_EMBEDDING_MODEL
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    vectors_db = FAISS.from_embeddings(split_chunks,embeddings)
    vectors_db.save_local(VECTOR_DB_PATH)
    print("向量数据库创建并保存完成")
    return vectors_db

def load_vectors(use_deepseek_embedding = True):
    if use_deepseek_embedding:
        embeddings = OpenAIEmbeddings(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model_name=DEEPSEEK_EMBEDDING_MODEL
        )
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = FAISS.load_local(VECTOR_DB_PATH,embeddings,allow_pickle=True)
    print("向量数据库加载完成")
    return vector_db

def retrieval(question,split_db,use_deepseek=True):
    retriever = split_db.as_retriever(search_kwargs={"K":TOP_K})
    relevant_chunks = retriever.get_relevant_documents(question)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    print(f"\n检索到的相关上下文（Top-{TOP_K}）：\n{context}\n")

    prompt = f"""
    请严格根据以下上下文信息回答用户的问题，仅使用上下文内容，不要编造任何信息。
    如果上下文没有相关答案，请直接回答“无法从文档中找到相关答案”。
    
    上下文：
    {context}
    
    用户问题：
    {question}
    """

    if use_deepseek:
        llm = ChatOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model_name=DEEPSEEK_EMBEDDING_MODEL,
            temperature = 0,
            max_tokens = 1024
        )
        answer = llm.invoke(prompt).content
    else:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer
        model_name = "Qwen/Qwen-7B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        pipe = pipeline("text-generation",model = model,tokenizer = tokenizer,max_new_tokens = 1024)
        llm = HuggingFacePipeline(pipeline=pipe)
        answer = llm.invoke(prompt)

    return answer

if __name__ == "__main__":
    USE_DEEPSEEK_EMBEDDING = True
    USE_DEEPSEEK_LLM = True

    chunks = chunking(PDF_PATH)
    db = create_vectors(chunks,use_deepseek_embedding = USE_DEEPSEEK_EMBEDDING)
    #db = load_vectors(USE_DEEPSEEK_EMBEDDING)

    while True:
        question = input("\n请输入你的问题（输入 'exit' 退出）：")
        if question.lower() == "exit":
            break
        answer = retrieval(question,db,USE_DEEPSEEK_LLM)
        print(f"\n回答:{answer}")
