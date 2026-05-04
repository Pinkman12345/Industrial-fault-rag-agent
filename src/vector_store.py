from typing import List
import hashlib
import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from config import settings
from document_loader import load_and_split_documents


class LocalHashEmbeddings(Embeddings):
    """
    一个本地简易 Embedding 实现。

    作用：
    - 不依赖外部 API；
    - 不需要下载模型；
    - 用于开发阶段跑通 Chroma 向量库流程。

    注意：
    - 这不是高质量语义向量模型；
    - 后续可以替换为 OpenAIEmbeddings、Qwen Embedding 或 bge 本地模型。
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vector = np.zeros(self.dim, dtype=np.float32)

        # 简单按字符和片段做哈希，适合中文关键词检索的开发测试
        tokens = list(text)

        # 加入一些连续字符片段，增强中文短语匹配能力
        for i in range(len(text) - 1):
            tokens.append(text[i:i + 2])

        for i in range(len(text) - 2):
            tokens.append(text[i:i + 3])

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = h % self.dim
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_model() -> Embeddings:
    """
    获取 Embedding 模型。

    第一版默认使用 LocalHashEmbeddings，保证项目在没有 API Key 的情况下也能跑通。
    后续可以在这里替换为真正的语义 Embedding 模型。
    """

    return LocalHashEmbeddings(dim=384)


def build_vector_store() -> Chroma:
    """
    构建 Chroma 向量库。

    流程：
    1. 加载并切分 docs/ 下的文档；
    2. 使用 Embedding 模型向量化；
    3. 存入本地 Chroma 数据库。
    """

    print("正在加载并切分文档...")
    chunks = load_and_split_documents()
    print(f"文档片段数量: {len(chunks)}")

    embedding_model = get_embedding_model()

    print("正在构建 Chroma 向量库...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(settings.VECTOR_DB_DIR)
    )

    vector_store.persist()
    print(f"向量库已保存到: {settings.VECTOR_DB_DIR}")

    return vector_store


def load_vector_store() -> Chroma:
    """
    加载已经构建好的 Chroma 向量库。
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(settings.VECTOR_DB_DIR),
        embedding_function=embedding_model
    )

    return vector_store


def search_similar_documents(query: str, top_k: int = 4) -> List[Document]:
    """
    根据用户问题检索相关文档片段。
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=top_k
    )

    return results


if __name__ == "__main__":
    # 第一次运行时，先构建向量库
    build_vector_store()

    print("\n" + "=" * 80)
    print("开始测试检索")
    print("=" * 80)

    test_query = "推进速度下降，同时刀盘转矩升高，可能是什么原因？"

    results = search_similar_documents(test_query, top_k=4)

    print(f"测试问题: {test_query}")
    print(f"检索结果数量: {len(results)}")
    print("-" * 80)

    for i, doc in enumerate(results, start=1):
        print(f"检索结果 {i}")
        print("来源文件:", doc.metadata.get("source"))
        print("片段编号:", doc.metadata.get("chunk_id"))
        print("内容预览:")
        print(doc.page_content[:500])
        print("-" * 80)