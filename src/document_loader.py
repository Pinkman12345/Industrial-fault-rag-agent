from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings


def load_markdown_documents(docs_dir: Path = settings.DOCS_DIR) -> List[Document]:
    """
    加载 docs 目录下的所有 Markdown 文档，并转换为 LangChain Document 对象。
    """

    documents: List[Document] = []

    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    markdown_files = list(docs_dir.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(f"未在目录中找到 Markdown 文档: {docs_dir}")

    for file_path in markdown_files:
        content = file_path.read_text(encoding="utf-8")

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "path": str(file_path)
            }
        )

        documents.append(doc)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    对文档进行切分，生成适合向量检索的小文本块。
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "，", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


def load_and_split_documents() -> List[Document]:
    """
    一步完成：加载 Markdown 文档并切分。
    """

    documents = load_markdown_documents()
    chunks = split_documents(documents)
    return chunks


if __name__ == "__main__":
    docs = load_markdown_documents()
    chunks = split_documents(docs)

    print(f"原始文档数量: {len(docs)}")
    print(f"切分后片段数量: {len(chunks)}")
    print("-" * 60)

    for i, chunk in enumerate(chunks[:5]):
        print(f"片段 {i}")
        print("来源文件:", chunk.metadata.get("source"))
        print("片段编号:", chunk.metadata.get("chunk_id"))
        print("内容预览:")
        print(chunk.page_content[:300])
        print("-" * 60)