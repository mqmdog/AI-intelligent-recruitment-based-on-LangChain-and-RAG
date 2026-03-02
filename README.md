# MQMDOG - 基于 LangChain 和 RAG 的 AI 智能招聘系统

一个基于 **LangChain** 框架和 **RAG (检索增强生成)** 技术构建的全栈智能招聘管理系统。系统通过多层 RAG 架构实现求职者智能问答，利用 LangChain Agent 自动化候选人评分与面试安排，显著提升招聘效率。

## 核心设计

- **四层 RAG 架构**: 查询理解 → 混合检索 → LLM 重排序 → 流式生成，实现精准的职位智能问答
- **LangChain 多工具 Agent**: 自动化候选人评分、面试邀请、日程协调的端到端招聘流程
- **LangGraph 状态管理**: 基于 PostgreSQL Checkpointer 的持久化 Agent 状态，支持多轮异步交互
- **混合检索引擎**: PgVector 向量搜索 + PostgreSQL 全文检索，RRF 融合排序
- **流式 SSE 响应**: 实时 Token 推送生成结果，前端即时展示

## 系统架构

系统采用前后端分离架构，前端 Vue 3 应用通过 HTTP/SSE 协议与后端 FastAPI 服务通信。后端核心由 AI 层（RAG 问答引擎 + LangChain Agent 集群）和集成层（钉钉 API、邮件服务、OCR 解析）组成，数据层采用 PostgreSQL 主库（业务数据）与 AI 专用库（向量数据 + Agent 状态）分离设计，Redis 作为缓存中间件。

```
前端 (Vue 3)
├── 职位管理模块
├── 候选人管理模块
└── 求职者智能问答模块 (SSE 流式)

后端 (FastAPI)
├── AI 层
│   ├── 四层 RAG 问答引擎
│   │   ├── Layer 1: 查询理解与改写 (LLM)
│   │   ├── Layer 2: 混合检索 (Vector + Keyword)
│   │   ├── Layer 3: LLM 重排序与过滤
│   │   └── Layer 4: 流式生成与验证
│   ├── 候选人评分 Agent (多工具编排)
│   └── 简历提取 Agent (结构化输出)
├── 集成层
│   ├── 钉钉集成模块
│   ├── 邮件机器人
│   └── OCR 解析服务
└── 数据访问层

数据存储
├── PostgreSQL 主库 (业务数据)
├── PostgreSQL AI 库 (向量数据 + Agent 状态)
└── Redis 缓存
```

## RAG 架构详解

系统采用四层递进式 RAG 架构处理求职者的自然语言查询：

### Layer 1 - 查询理解与改写

利用 LLM 分析用户原始问题，识别查询意图并生成多个优化后的检索查询。例如用户询问"有没有薪资比较高的后端开发岗位？"，系统会识别意图为薪资相关的职位搜索，生成"高薪资 后端开发 职位"、"后端工程师 薪酬待遇"等多个改写查询，并提取"后端开发"、"高薪资"作为关键实体。

系统支持 6 种意图分类：职位搜索、薪资查询、任职要求、部门查询、截止日期、通用咨询。

### Layer 2 - 混合检索引擎

系统同时执行向量语义搜索和关键词全文检索，通过 RRF 算法融合两路检索结果。向量搜索使用 DashScope text-embedding-v3 模型将查询转化为 1024 维向量，在 PgVector 中执行余弦相似度检索；关键词搜索利用 PostgreSQL 内置的全文检索能力。RRF 融合算法对两路检索结果按排名倒数加权，兼顾语义相关性和关键词精确匹配，最终输出 Top-10 候选文档。

### Layer 3 - LLM 重排序与过滤

使用 DeepSeek 模型对候选文档进行 1-10 分的相关性评分，仅保留评分不低于 3 分的文档。系统还会对同一职位进行去重处理，确保每个职位最多保留一个文本块，最终输出 Top-5 高相关文档。

### Layer 4 - 流式生成与验证

基于检索到的职位信息构建结构化上下文，通过 LLM 流式生成回答。系统严格基于检索文档回答，杜绝信息编造，并通过 SSE 协议实时逐 Token 推送结果到前端。对话历史和检索来源自动保存，支持多轮上下文理解。

## LangChain Agent 详解

### 候选人评分与面试安排 Agent

基于 LangChain 构建的多工具协作 Agent，实现自动化招聘流程。Agent 核心使用 Qwen3-MAX 模型，通过多个工具协同完成从候选人评分到面试确认的全流程：

- **评分工具**: 从工作经验（30%）、技术技能（30%）、项目经验（20%）、软技能（10%）、教育背景（10%）五个维度对候选人进行 AI 评估，总分超过 4 分自动进入下一轮
- **邮件邀请工具**: 向候选人发送协商面试时间的邀请邮件
- **日程查询工具**: 通过钉钉 API 获取面试官未来 7 天的空闲时间段
- **确认面试工具**: 执行最终确认的四步操作（发送确认邮件、创建钉钉日程、生成面试记录、更新候选人状态）
- **拒绝处理工具**: 更新候选人状态为拒绝面试
- **时间工具**: 提供当前时间信息辅助决策

**关键技术特性:**

- **模型降级**: 通过 ModelFallbackMiddleware 实现主模型失败时自动切换备用模型
- **长对话摘要**: SummarizationMiddleware 在对话超过 50000 tokens 时自动压缩历史
- **状态持久化**: AsyncPostgresSaver 将 Agent 执行状态存入 PostgreSQL
- **工具状态共享**: ToolRuntime 实现工具函数间共享候选人、职位、面试官信息

### 简历信息提取 Agent

利用 LangChain 结构化输出能力，从 OCR 解析的简历文本中精准提取候选人信息，流程为：简历 PDF 上传 → OCR 文字识别 → LLM Agent 分析 → 输出符合 Pydantic Schema 的结构化 JSON 数据，包含姓名、邮箱、电话、教育背景、工作经历、技能标签等字段。

## 向量化存储

职位信息按 4 种维度分块向量化，存储于 PgVector：

- **TITLE 类型**: 职位标题，用于标题级语义匹配
- **DESCRIPTION 类型**: 职位描述，用于职责内容检索
- **REQUIREMENTS 类型**: 任职要求，用于技能经验匹配
- **FULL 类型**: 完整信息，用于全局语义理解

## 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.124 | 异步 Web 框架 |
| LangChain | 1.2.0 | LLM 应用框架 |
| LangGraph | 1.0.5 | Agent 状态图与编排 |
| LangChain-OpenAI | 1.1.4 | LLM 接口适配 |
| SQLAlchemy | 2.0.45 | 异步 ORM |
| PgVector | 0.1.8 | PostgreSQL 向量扩展 |
| Pydantic | 2.12 | 数据校验与序列化 |
| Redis | 7.1.0 | 缓存与状态存储 |
| Alembic | 1.17 | 数据库迁移 |

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue 3 | 3.5.25 | 前端框架 |
| TypeScript | 5.9 | 类型安全 |
| Vite | 7.3 | 构建工具 |
| Element Plus | 2.13 | UI 组件库 |
| ECharts | 6.0 | 数据可视化 |
| Pinia | 3.0 | 状态管理 |

### AI 模型

| 模型 | 用途 |
|------|------|
| 通义千问 Qwen3-MAX | 主力 LLM，用于问答生成、Agent 推理、信息提取 |
| DeepSeek v3.2 | 备用 LLM 及文档重排序 |
| DashScope text-embedding-v3 | 文本向量化，输出 1024 维向量 |
| PaddleOCR | 简历 PDF/图片文字识别 |

## 项目结构

```
hr-system/
├── hr-backend/
│   ├── agents/                    # AI Agent 实现
│   │   ├── llms.py                # LLM 配置 (Qwen + DeepSeek)
│   │   ├── candidate.py           # 候选人评分与面试 Agent
│   │   ├── resume.py              # 简历信息提取 Agent
│   │   └── job_qa_agent.py        # 四层 RAG 问答引擎
│   ├── core/
│   │   ├── embedding.py           # DashScope 向量化客户端
│   │   ├── auth.py                # JWT 认证
│   │   ├── dingtalk.py            # 钉钉 API 集成
│   │   ├── mail.py                # 邮件服务
│   │   ├── ocr.py                 # OCR 服务
│   │   └── email_bot/             # 邮件机器人 (IMAP/SMTP)
│   ├── models/                    # SQLAlchemy 数据模型
│   ├── repository/                # 数据访问层
│   │   └── position_embedding_repo.py  # 向量存储与检索
│   ├── routers/                   # API 路由
│   ├── schemas/                   # Pydantic 数据模式
│   ├── scheduler/                 # APScheduler 定时任务
│   ├── settings/                  # 配置管理
│   ├── alembic/                   # 数据库迁移
│   └── main.py                    # 应用入口
├── hr-frontend/
│   └── src/
│       ├── views/
│       │   ├── seeker/
│       │   │   ├── HomeView.vue   # 智能问答页面 (SSE 流式)
│       │   │   └── ApplyView.vue  # 简历投递页面
│       │   └── internal/
│       │       ├── DashboardView.vue   # 数据仪表板
│       │       ├── PositionsView.vue   # 职位管理
│       │       └── CandidatesView.vue  # 候选人管理
│       ├── stores/
│       │   └── chat.ts            # 对话状态管理
│       ├── utils/
│       │   └── sse.ts             # SSE 流式连接工具
│       └── api/                   # API 调用层
└── README.md
```

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ (需安装 pgvector 扩展)
- Redis

### 数据库准备

```sql
-- 创建数据库
CREATE DATABASE hr_system;
CREATE DATABASE hr_system_agent;

-- 安装 pgvector 扩展
\c hr_system
CREATE EXTENSION IF NOT EXISTS vector;
\c hr_system_agent
CREATE EXTENSION IF NOT EXISTS vector;
```

### 后端部署

```bash
cd hr-backend

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (复制模板并填写实际值)
cp .env.example .env

# 数据库迁移
alembic upgrade head

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 前端部署

```bash
cd hr-frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build
```

### 环境变量配置

参考 `hr-backend/.env.example`：

```env
# 数据库
DB_PASSWORD=your_db_password

# JWT
JWT_SECRET_KEY=your_jwt_secret_key

# 邮箱 (QQ 邮箱)
MAIL_USERNAME=your_email@qq.com
MAIL_PASSWORD=your_email_password


# OCR 服务
PADDLE_OCR_ACCESS_TOKEN=your_paddle_ocr_access_token

# AI 大模型 (阿里云 DashScope)
DASHSCOPE_API_KEY=your_dashscope_api_key
```

## 功能模块

### 求职者端

- **智能问答**: 基于 RAG 的职位咨询，支持流式回答和多轮对话，可查询职位详情、薪资范围、任职要求等信息
- **简历投递**: 上传简历 → OCR 解析 → AI 信息提取 → 一键投递，全程自动化处理

### 管理端

- **职位管理**: 发布、编辑、删除职位，支持一键向量化入库，可设置职位状态、薪资范围、学历要求等
- **候选人管理**: AI 自动评分，实时跟踪候选人状态，支持面试安排和结果记录
- **数据仪表板**: 招聘数据统计与可视化，包括职位发布量、投递量、通过率等核心指标
- **面试协调**: 自动查询面试官日历，发送邮件邀请，实现全流程自动化

## 许可证

本项目基于 MIT License 开源。
