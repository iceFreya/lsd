import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter)

TEST_MARKDOWN = """
# AI大模型生产环境部署与运维全指南

## 1. 基础环境配置要求
### 1.1 硬件规格
* 推理节点：4 × NVIDIA H100 80GB GPU
* 内存：1TB ECC 服务器内存
* 存储：4TB NVMe 固态，用于模型权重与缓存
### 1.2 软件依赖
系统必须使用 Ubuntu 22.04 LTS，预装 CUDA 12.4、cuDNN 9.0
所有服务禁止使用 root 账户启动，必须创建专用服务用户 llm-service

## 2. Docker 容器化部署
### 2.1 部署命令
执行以下命令启动模型服务，必须挂载本地模型目录与日志目录：
```bash
# 拉取官方镜像
docker pull llm-registry/prod-model:v3.1
# 启动容器（核心参数）
docker run -d \
  --gpus all \
  -p 9000:9000 \
  -v /data/models:/app/models \
  -v /data/logs:/app/logs \
  --name llm-server \
  llm-registry/prod-model:v3.1
```
### 2.2 健康检查
服务启动后，访问 http://localhost:9000/health 验证状态，返回 status:ok 即为成功。
## 3. 性能优化方案
### 3.1 显存优化
当模型显存占用超过 85% 时，必须启用 INT8 量化，可降低 50% 显存占用。
同时关闭冗余的梯度计算功能，进一步释放硬件资源。
### 3.2 并发优化
调整环境变量 MAX_BATCH_SIZE=32，提升单节点并发处理能力，
优化后单秒请求吞吐量可提升 3 倍。
## 4. 监控与告警规则
### 4.1 核心监控指标
使用 Prometheus + Grafana 监控：GPU 使用率、显存占用、接口响应时间。
### 4.2 告警触发条件
显存占用持续 5 分钟 > 90%，触发一级告警
接口响应时间 > 500ms，触发二级告警
### 4.3 告警处理流程
一级告警优先执行 INT8 量化热更新，无需重启服务；
二级告警需检查并发参数是否超限，及时调整批量大小。
## 5. 安全合规规范
所有对外接口必须开启 API Key 鉴权，禁止匿名访问；
服务日志仅保留 180 天，满足企业等保三级要求；
严禁将用户隐私数据输入模型进行实时推理。

"""

def naive_split(text,chunk_size:int = 150,chunk_overlap:int = 20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n"," ",""],
    )
    naive_chunks = splitter.split_text(text)
    return naive_chunks

def semantic_split(text):
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#","Header 1"),
            ("##","Header 2"),
            ("###","Header 3"),
        ]
    )
    markdown_text = markdown_splitter.split_text(text)

    recursive_spliter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n","\n"," ",""],
    )
    semantic_chunks = recursive_spliter.split_documents(markdown_text)

    return semantic_chunks

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

chunks_n = naive_split(TEST_MARKDOWN)
chunks_s = semantic_split(TEST_MARKDOWN)

db_n = FAISS.from_texts(chunks_n,embeddings)
db_s = FAISS.from_documents(chunks_s,embeddings)

test_query = "显存占用过高的告警触发条件是什么？对应的处理方案有哪些？"

print(f"测试问题:{test_query}")

top_chunk_a = db_n.similarity_search(test_query, k=1)[0]
top_chunk_b = db_s.similarity_search(test_query, k=1)[0]

print("\n【策略 A （Naive） 检索结果】：")
print(f"内容： {top_chunk_a.page_content.strip()}")


print("\n【策略 B （Semantic） 检索结果】：")
print(f"内容： {top_chunk_b.page_content.strip()}")
print(f"携带的元数据 （Metadata）： {top_chunk_b.metadata}")


