from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from schemas.agent_schema import AgentCandidateSchema

from .llms import qwen_llm, deepseek_llm
from .prompts import EXTRACT_CANDIDATE_SYSTEM_PROMPT

agent = create_agent(
    model=qwen_llm,
    middleware=[ModelFallbackMiddleware(first_model=deepseek_llm)],
    system_prompt=EXTRACT_CANDIDATE_SYSTEM_PROMPT,
    response_format=AgentCandidateSchema,
)


# 输入OCR解析内容，输出候选人信息
async def extract_candidate_info(content: str) -> AgentCandidateSchema:
    response = await agent.ainvoke({"messages": [{
        "role": "user",
        "content": f"原始文本内容为：{content}，请从原始文本中提取候选人信息成JSON格式",
    }]})
    return response['structured_response']
