import json
from datetime import datetime, timedelta
from typing import Annotated, Any, List, Optional, TypeVar

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, SummarizationMiddleware #模型降级中间件，总结中间件
from langchain.tools import tool, ToolRuntime  # 工具运行时上下文，提供状态访问
from langchain_core.prompts import PromptTemplate #提示词模版
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import BaseMessage, add_messages
from loguru import logger
from pydantic import BaseModel

from core.cache import HRCache
from core.dingtalk import DingTalkHttp
from core.email_bot import EmailBot, EmailBotSettings
from models import AsyncSessionFactory
from models.candidate import CandidateStatusEnum
from models.interview import InterviewResultEnum
from repository.candidate_repo import CandidateRepo
from repository.candidate_score_repo import CandidateAIScoreRepo
from repository.interview_repo import InterviewRepo
from repository.user_repo import UserRepo
from schemas.agent_schema import AgentCandidateScoreSchema
from schemas.cache_schema import DingTalkTokenInfoSchema
from schemas.candidate_schema import CandidateSchema
from schemas.position_schema import PositionSchema
from schemas.user_schema import UserSchema
from settings import settings
from utils.available_time import find_available_slot
from utils.iso8601 import datetime_to_iso8601_beijing, iso8601_to_datetime_beijing

from .llms import qwen_llm, deepseek_llm
from .prompts import (
    CANDIDATE_PROCESS_SYSTEM_PROMPT,
    SCORE_FOR_CANDIDATE_SYSTEM_PROMPT,
    SCORE_FOR_CANDIDATE_USER_PROMPT,
)


async def get_dingtalk_access_token(user_id: str) -> str:
    """通过缓存中的 refresh_token 刷新并返回钉钉 access_token"""
    dingding_http = DingTalkHttp()
    cache: HRCache = HRCache()
    token_info = await cache.get_dingtalk_info(user_id)
    if not token_info:
        error_message = f"{user_id}用户钉钉授权已过期！"
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        refresh_token, access_token = await dingding_http.refresh_access_token(token_info.refresh_token)
        await cache.set_dingtalk_token_info(
            DingTalkTokenInfoSchema(
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token
            )
        )

        return access_token
    except ValueError as e:
        error_str = str(e)
        if "invalidParameter.authCode.notFound" in error_str or "authCode" in error_str:
            logger.warning(f"用户 {user_id} 的钉钉 authCode 已失效，清除缓存并提示用户重新授权")
            cache_key = f'{cache.dingtalk_prefix}{user_id}'
            await cache.delete(cache_key)
            raise ValueError(
                "钉钉授权已过期，请重新授权。请调用 /user/dingtalk/reauthorize 清除旧授权，"
                "然后调用 /user/dingtalk/authorize 重新授权。"
            )
        logger.error(f"Token 刷新失败：{error_str}")
        raise
    except Exception as e:
        logger.error(f"获取钉钉 access_token 异常：{str(e)}")
        raise ValueError(f"获取钉钉 access_token 失败：{str(e)}")


T = TypeVar("T")

# 状态属性赋值
def assign_state_property(left: T, right: Optional[T]) -> T:
    return right if right is not None else left


class CandidateAgentState(BaseModel):
    """候选人处理 Agent 的状态，checkpointer 会自动持久化"""
    message: Annotated[List[BaseMessage], add_messages]
    candidate: Annotated[CandidateSchema, assign_state_property]
    position: Annotated[PositionSchema, assign_state_property]
    interviewer: Annotated[UserSchema, assign_state_property]
# Annotated[类型, 元数据]：为类型添加额外元数据


@tool
async def score_for_candidate(
        runtime: ToolRuntime[CandidateAgentState]
):
    """
    根据职位信息，给职位上的候选人进行评分。
    评分结果会存入数据库中，并根据评分结果修改候选人状态。
    return：
    - str | None: 评分结果的JSON字符串，如果评分失败则返回None
    """
    candidate: CandidateSchema = runtime.state['candidate'] 
    position: PositionSchema = runtime.state['position']
    #1.创建评分 Agent
    score_agent = create_agent(
        model=qwen_llm,
        system_prompt=SCORE_FOR_CANDIDATE_SYSTEM_PROMPT,
        middleware=[
            ModelFallbackMiddleware(
                first_model=deepseek_llm
            ),
        ],
        response_format=AgentCandidateScoreSchema
    )
    #2.创建用户提示
    user_prompt_template = PromptTemplate.from_template(
        SCORE_FOR_CANDIDATE_USER_PROMPT)
    user_prompt = user_prompt_template.invoke(
        {"candidate": candidate.model_dump_json(), "position": position.model_dump_json()})
    #3.调用Agent
    response = await score_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": user_prompt.text
        }]
    })
    candidate_score: AgentCandidateScoreSchema = response['structured_response']
    #4.保存评分结果
    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                score_repo = CandidateAIScoreRepo(session)
                candidate_repo = CandidateRepo(session)
                await score_repo.create_candidate_score(
                    candidate_id=candidate.id,
                    candidate_score_dict=candidate_score.model_dump()
                )
                status = CandidateStatusEnum.AI_FILTER_FAILED
                if candidate_score.overall_score > 4:
                    status = CandidateStatusEnum.AI_FILTER_PASSED
                await candidate_repo.update_candidate_status(candidate_id=candidate.id, status=status)
            except Exception as e:
                return f'得分工具执行失败，错误信息为：{e}'
    return f'得分工具执行成功，该候选人得分情况为：{candidate_score.model_dump_json()}'


@tool
async def get_interviewer_available_slot(
    runtime: ToolRuntime[CandidateAgentState]
):
    """
    根据职位信息，获取面试官可用的面试时间。
    """
    interviewer: UserSchema = runtime.state['interviewer']
    union_id: str | None = None
    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                user_repo = UserRepo(session)
                dingding_user = await user_repo.get_dingding_user(
                    user_id=interviewer.id)
                if not dingding_user:
                    return "获取面试官可用时间失败，面试官没有绑定钉钉账号！"
                union_id = dingding_user.union_id
            except Exception as e:
                return f"获取面试官可用时间失败：{e}"

    try:
        access_token: str = await get_dingtalk_access_token(interviewer.id)
    except Exception as e:
        logger.error(e)
        return f"获取面试官可用时间失败：{e}"

    # 从钉钉获取面试官日程安排
    try:
        dingtalk_http = DingTalkHttp()
        now = datetime.now()
        tomorrow_nine = datetime(
            year=now.year, month=now.month, day=now.day, hour=9)
        events: list[dict[str, Any]] = await dingtalk_http.get_calendar_list(
            union_id=union_id,
            access_token=access_token,
            time_min=tomorrow_nine,
            time_max=tomorrow_nine + timedelta(days=7),
        )
        busy_slots = [
            (iso8601_to_datetime_beijing(
                event['start']['dateTime']), iso8601_to_datetime_beijing(event['end']['dateTime']))
            for event in events
        ]
        available_slots: list[tuple[datetime, datetime]] = find_available_slot(
            busy_slots,
            start_date=tomorrow_nine
        )
        if len(available_slots) == 0:
            return "获取面试官可用时间失败：7天内没有空闲时间！"
        available_times = [
            (datetime_to_iso8601_beijing(slot[0]),
             datetime_to_iso8601_beijing(slot[1]))
            for slot in available_slots
        ]
        return f"找到面试官可用的时间：{json.dumps(available_times)}"
    except Exception as e:
        return f"获取面试官可用时间失败：{e}"


@tool
async def send_interview_email(
    interview_datetime_str: str,
    runtime: ToolRuntime[CandidateAgentState],
):
    """
    给候选人发送面试时间邀请（并非最终面试时间，后续可能还需要通过邮件来和候选人协商最终面试时间）
    :param interview_datetime_str: 面试时间的字符串
    """
    candidate = runtime.state['candidate']
    position = runtime.state['position']

    email_bot_settings = EmailBotSettings(
        imap_host=settings.EMAIL_BOT_IMAP_HOST,
        smtp_host=settings.EMAIL_BOT_SMTP_HOST,
        email=settings.EMAIL_BOT_EMAIL,
        password=settings.EMAIL_BOT_PASSWORD,
    )
    async with EmailBot(email_bot_settings) as bot:
        subject = "MQQDOG面试邀请-协商面试时间"
        body = f"""
尊敬的{candidate.name}，
您好！
感谢您投递我司{position.title}职位。
我们初步确定了您的面试时间，请您确认是否方便。
面试时间：{interview_datetime_str}
请您确认是否方便，如果方便，请您回复“确认”。
如果不方便，请回复您方便的时间，我们将会重新协商面试时间。
谢谢！
        """
        try:
            await bot.send_email(
                to=candidate.email,
                subject=subject,
                text=body,
            )
        except Exception as e:
            logger.error(e)
            return f"给候选人发送邮件失败：{e}"

        return f"给候选人发送面试邀请邮件成功！面试时间初步确定为：{interview_datetime_str}"


@tool
async def confirm_interview_time(
    interview_datetime_str: str,
    runtime: ToolRuntime[CandidateAgentState],
):
    """
    确认最终的面试时间。这个工具会做以下四件事：
    * 通过邮件，发送最终确认面试的时间给候选人
    * 给面试官的钉钉创建一个面试的日程安排
    * 在系统中创建一个面试预约记录
    # 在系统中修改候选人的状态为待面试
    :param interview_datetime_str: 面试时间，ISO8601格式的字符串
    """
    position = runtime.state['position']
    candidate = runtime.state['candidate']
    interviewer = runtime.state['interviewer']

    try:
        interview_datetime: datetime = iso8601_to_datetime_beijing(
            interview_datetime_str)
        # 由于后面存储数据库，不能在日期中带时区，所以这里先处理一下
        if interview_datetime.tzinfo is not None:
            interview_datetime_without_tz = interview_datetime.astimezone(
                None).replace(tzinfo=None)
        else:
            interview_datetime_without_tz = interview_datetime
    except Exception as e:
        return f"{interview_datetime_str}格式化失败！{e}"

    # 1. 通过邮件，发送最终确认面试的时间给候选人
    email_bot_settings = EmailBotSettings(
        imap_host=settings.EMAIL_BOT_IMAP_HOST,
        smtp_host=settings.EMAIL_BOT_SMTP_HOST,
        email=settings.EMAIL_BOT_EMAIL,
        password=settings.EMAIL_BOT_PASSWORD,
    )
    async with EmailBot(email_bot_settings) as bot:
        subject = "MQMDOG面试时间确定"
        body = f"""
        "尊敬的{candidate.name}，
        面试时间已确定：
        {interview_datetime_str}
        请您准时参加面试。该邮件无需再回复。谢谢！
        """
        try:
            await bot.send_email(
                to=candidate.email,
                subject=subject,
                text=body,
            )
        except Exception as e:
            logger.error(e)
            return f"给候选人发送邮件失败：{e}"

    # 2. 给面试官的钉钉创建一个面试的日程安排
    dingtalk_error_msg = None
    union_id: str | None = None
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                user_repo = UserRepo(session)
                dingding_user = await user_repo.get_dingding_user(user_id=interviewer.id)
                if not dingding_user:
                    dingtalk_error_msg = "面试官没有绑定钉钉账号！无法创建日程安排！"
                else:
                    union_id = dingding_user.union_id
    except Exception as e:
        dingtalk_error_msg = f"面试官用户信息获取失败！{e}"

    # 如果获取到了union_id，则尝试创建日程
    if union_id:
        try:
            access_token: str = await get_dingtalk_access_token(interviewer.id)
            dingtalk_http = DingTalkHttp()
            end_datetime: datetime = interview_datetime + timedelta(hours=1)
            logger.info(
                f"[钉钉日程创建] 准备创建日程 - union_id={union_id}, summary={position.title} - {candidate.name}, start={interview_datetime}, end={end_datetime}")
            await dingtalk_http.create_calendar(
                union_id=union_id,
                access_token=access_token,
                summary=f"面试安排：{position.title} - {candidate.name}",
                start_datetime=interview_datetime,
                end_datetime=end_datetime,
            )
            logger.info(f"[钉钉日程创建] 成功为面试官 {interviewer.id} 创建日程")
        except Exception as e:
            # 钉钉日程创建失败，但不阻断整个流程
            logger.error(
                f"[钉钉日程创建] 失败 - 错误类型: {type(e).__name__}, 错误信息: {str(e)}")
            logger.warning(f"给面试官创建钉钉日程失败，但继续保存数据库记录：{e}")
            dingtalk_error_msg = f"给面试官创建钉钉日程失败：{e}"

    try:
        async with AsyncSessionFactory() as session:
            interview_repo = InterviewRepo(session)
            # 3. 在数据库中创建一个面试预约记录
            await interview_repo.create_interview({
                # 面试时间因为是存储到postgresql数据库中，现在是只能存储没有带时区的日期
                "scheduled_time": interview_datetime_without_tz,
                "result": InterviewResultEnum.PENDING,
                "candidate_id": candidate.id,
                "interviewer_id": interviewer.id,
            })
            # 4. 在数据库中修改候选人的状态为待面试
            candidate_repo = CandidateRepo(session)
            await candidate_repo.update_candidate_status(
                candidate_id=candidate.id,
                status=CandidateStatusEnum.WAITING_FOR_INTERVIEW,
            )
            # 手动提交事务，确保数据被永久化
            await session.commit()
    except Exception as e:
        return f"在系统中创建面试预约记录和候选人状态修改失败！{e}"

    return f"""
    * 给候选人发送面试时间执行成功！
    * 给面试官创建钉钉日程安排成功！
    * 在系统中创建面试预约成功！
    * 在系统中修改候选人状态为待面试成功！
    """


@tool
async def refuse_interview(
    runtime: ToolRuntime[CandidateAgentState],
):
    """
    如果候选人拒绝了面试，那么调用该工具来更新候选人状态为拒绝面试
    """
    candidate = runtime.state['candidate']
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                candidate_repo = CandidateRepo(session)
                await candidate_repo.update_candidate_status(
                    candidate_id=candidate.id,
                    status=CandidateStatusEnum.REFUSED_INTERVIEW,
                )
        return "已修改候选人状态为拒绝面试！"
    except Exception as e:
        return f"修改候选人状态为拒绝面试失败！{e}"


@tool
def get_current_time() -> str:
    """
    获取当前时间（北京时间）：年月日、时分秒、星期几、当月第几天
    """
    now_bj = datetime.now()

    weekday_map = {
        0: "星期一",
        1: "星期二",
        2: "星期三",
        3: "星期四",
        4: "星期五",
        5: "星期六",
        6: "星期日",
    }
    weekday_cn = weekday_map[now_bj.weekday()]

    day_of_month = now_bj.day

    return (
        f"{now_bj.year}年{now_bj.month}月{now_bj.day}日 "
        f"{now_bj.hour:02d}:{now_bj.minute:02d}:{now_bj.second:02d} "
        f"{weekday_cn}（本月第{day_of_month}天）"
    )


class CandidateProcessAgent:
    def __init__(
            self,
            candidate: CandidateSchema | None = None,
            position: PositionSchema | None = None,
            interviewer: UserSchema | None = None
    ):
        self.candidate = candidate
        self.position = position
        self.interviewer = interviewer
        self._checkpointer = None

    async def ainvoke(self, messages: list[BaseMessage], thread_id: str):
        assert self._checkpointer is not None
        agent = create_agent(
            model=qwen_llm,
            system_prompt=CANDIDATE_PROCESS_SYSTEM_PROMPT,
            state_schema=CandidateAgentState,
            middleware=[
                ModelFallbackMiddleware(
                    first_model=deepseek_llm
                ),
                SummarizationMiddleware(
                    model=deepseek_llm,
                    trigger=("tokens", 50000),
                    keep=("tokens", 10000)
                )
            ],
            tools=[score_for_candidate, get_interviewer_available_slot, send_interview_email,
                   confirm_interview_time, refuse_interview, get_current_time],
            checkpointer=self._checkpointer,
        )
        response = await agent.ainvoke({"messages": messages, "candidate": self.candidate, "position": self.position, "interviewer": self.interviewer}, {"thread_id": thread_id})
        return response

    async def __aenter__(self):
        self._checkpoint_conn = AsyncPostgresSaver.from_conn_string(
            settings.DATABASE_AGENT_URL)
        self._checkpointer = await self._checkpoint_conn.__aenter__()
        await self._checkpointer.setup()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._checkpoint_conn.__aexit__(exc_type, exc_value, traceback)
