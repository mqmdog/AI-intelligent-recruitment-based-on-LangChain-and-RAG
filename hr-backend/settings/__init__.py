from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field, Field
import os
from datetime import timedelta

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),  # 从 .env 文件加载环境变量
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 数据库
    DB_USERNAME: str = "postgres"
    DB_PASSWORD: str = Field(..., validation_alias="DB_PASSWORD")
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 5433
    DB_NAME: str = "hr_system"
    DB_AGENT_NAME: str = "hr_system_agent"

    # JWT
    JWT_SECRET_KEY: str = Field(..., validation_alias="JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(days=365)
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(days=365)

    # Redis
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    INVITE_CODE_EXPIRES: int = 60 * 60 * 24 * 2

    # 邮箱
    MAIL_USERNAME: str = Field(..., validation_alias="MAIL_USERNAME")
    MAIL_PASSWORD: str = Field(..., validation_alias="MAIL_PASSWORD")
    MAIL_FROM: str = Field(..., validation_alias="MAIL_USERNAME")
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.qq.com"
    MAIL_FROM_NAME: str = "MQMDOG"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False

    # 邮箱机器人
    EMAIL_BOT_IMAP_HOST: str = "imap.qq.com"
    EMAIL_BOT_SMTP_HOST: str = "smtp.qq.com"
    EMAIL_BOT_EMAIL: str = Field(..., validation_alias="MAIL_USERNAME")
    EMAIL_BOT_PASSWORD: str = Field(..., validation_alias="MAIL_PASSWORD")

    # 钉钉
    DINGTALK_CLIENT_ID: str = Field(..., validation_alias="DINGTALK_APP_KEY")
    DINGTALK_CLIENT_SECRET: str = Field(...,
                                        validation_alias="DINGTALK_APP_SECRET")

    # 后端域名，ngork穿透
    BACKEND_BASE_URL: str = "https://unreturnable-kaelyn-regularly.ngrok-free.dev/"

    # 简历上传存储路径
    RESUME_DIR: str = os.path.join(BASE_DIR, "upload")

    # OCR 服务
    PADDLE_OCR_ACCESS_TOKEN: str = Field(...,
                                         validation_alias="PADDLE_OCR_ACCESS_TOKEN")

    # AI 大模型
    DASHSCOPE_API_KEY: str = Field(..., validation_alias="DASHSCOPE_API_KEY")

    # RAG 配置
    EMBEDDING_MODEL: str = "text-embedding-v3"  # 通义千问的文本嵌入模型，用于将文本转换为向量
    EMBEDDING_DIMENSION: int = 1024  # 嵌入向量的维度
    EMBEDDING_BATCH_SIZE: int = 25  # 嵌入向量的批次大小
    RAG_TOP_K_VECTOR: int = 20  # 向量检索返回的最相似文档数量
    RAG_TOP_K_KEYWORD: int = 20  # 关键词检索返回的最相关文档数量
    RAG_TOP_K_RERANK: int = 5  # 最终重排后保留的文档数量

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg://{self.DB_USERNAME}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )  # 主数据库：存储业务数据、用户信息等核心数据

    @computed_field
    @property
    def DATABASE_AGENT_URL(self) -> str:
        return (
            f"postgresql://{self.DB_USERNAME}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_AGENT_NAME}"
        )  # AI数据库：存储向量数据、会话历史、RAG相关数据等

# 单例模式
settings = Settings()
