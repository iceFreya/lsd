import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_DIM = 384

parent_documents = [
    {"parent_id": 1, "content": "人工智能(AI)是一门模拟人类智能的技术。它包含机器学习、深度学习、自然语言处理等分支。机器学习是AI的核心，让计算机从数据中学习规律。"},
    {"parent_id": 2, "content": "RAG是检索增强生成技术，通过向量库检索外部知识，再交给LLM生成回答，解决大模型幻觉、知识过时问题。"},
    {"parent_id": 3, "content": "向量数据库专门用于存储和检索向量数据。在大模型应用中，它负责存储文本嵌入向量，实现语义搜索，是RAG系统的核心组件。"}
]

def child_chunks_spliter(parent_docs,chunk_size=15):
    temp_child_chunks = []
    for parent_doc in parent_docs:
        parent_id = parent_doc["parent_id"]
        content = parent_doc["content"]
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        for chunk in chunks:
            temp_child_chunks.append({"parent_id": parent_id, "child_content": chunk})
    return temp_child_chunks

child_chunks = child_chunks_spliter(parent_documents)

child_texts = [item["child_content"] for item in child_chunks]
child_embeddings = embed_model.encode(child_texts)

index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(np.array(child_embeddings).astype('float32'))

def parent_retriever(temp_query,top_k=3):
    query_embeddings = embed_model.encode([temp_query])
    distances, indices = index.search(np.array(query_embeddings).astype('float32'), top_k)

    parent_score = {}
    for idx,dist in zip(indices[0],distances[0]):
        parent_id = child_chunks[idx]["parent_id"]
        if parent_id not in parent_score or dist < parent_score[parent_id]:
            parent_score[parent_id] = dist

    sorted_parent_ids = sorted(parent_score.keys(),key=lambda x:parent_score[x])

    matched_parents = [p for p in parent_documents if p["parent_id"] in sorted_parent_ids]
    matched_parents.sort(key=lambda x:sorted_parent_ids.index(x["parent_id"]))
    return matched_parents

if __name__ == '__main__':
    query = "RAG技术是什么？"
    results = parent_retriever(query)
    print("检索结果:")
    for result in results:
        print(f"ID: {result['parent_id']}")
        print(f"content: {result['content']}")