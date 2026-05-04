import os
from pathlib import Path
from dotenv import load_dotenv


# 项目根目录：src/config.py 的上一级再上一级
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env 文件
load_dotenv(BASE_DIR / ".env")


class Settings:
    """
    项目全局配置类。
    统一管理 API、模型、路径等配置，避免后面代码里到处写硬编码。
    """

    # API 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # 路径配置
    DOCS_DIR: Path = BASE_DIR / "docs"
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db" / "chroma"

    # 文档切分配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80


settings = Settings()


if __name__ == "__main__":
    print("项目根目录:", BASE_DIR)
    print("文档目录:", settings.DOCS_DIR)
    print("数据目录:", settings.DATA_DIR)
    print("向量库目录:", settings.VECTOR_DB_DIR)
    print("LLM 模型:", settings.LLM_MODEL)
    print("Embedding 模型:", settings.EMBEDDING_MODEL)