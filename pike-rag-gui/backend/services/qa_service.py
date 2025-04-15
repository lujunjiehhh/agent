from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import QuestionRequest, QuestionResponse, ReasoningStep, Reference
from ..database import SessionLocal

class QAService:
    def __init__(self):
        self.db = SessionLocal()

    async def process_question(self, request: QuestionRequest) -> QuestionResponse:
        """处理问题并返回答案"""
        # TODO: 调用PIKE-RAG的问答功能
        # 这里需要集成PIKE-RAG的问答功能
        return QuestionResponse(
            answer="",
            reasoning_steps=[],
            references=[]
        )

qa_service = QAService() 