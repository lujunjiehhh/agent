from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from pathlib import Path
import shutil
import os
from datetime import datetime

from .models import (
    Document,
    DocumentResponse,
    KnowledgeGraph,
    QuestionRequest,
    QuestionResponse,
    ReasoningStep
)
from .database import get_db, init_db
from .services import (
    document_service,
    knowledge_graph_service,
    qa_service
)

app = FastAPI(title="PIKE-RAG GUI API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建上传目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档并处理"""
    try:
        # 保存文件
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 处理文档
        doc = await document_service.process_document(file_path)
        return DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            created_at=doc.created_at,
            processed_at=doc.processed_at
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents():
    """获取所有文档列表"""
    return await document_service.get_all_documents()

@app.get("/api/documents/{doc_id}/preview")
async def preview_document(doc_id: str):
    """预览文档内容"""
    try:
        content = await document_service.get_document_preview(doc_id)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/qa", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """处理问题并返回答案"""
    try:
        response = await qa_service.process_question(request.question)
        return QuestionResponse(
            answer=response.answer,
            reasoning_steps=response.reasoning_steps,
            references=response.references
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/knowledge-graph", response_model=KnowledgeGraph)
async def get_knowledge_graph():
    """获取知识图谱"""
    try:
        return await knowledge_graph_service.get_graph()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/knowledge-graph/entities/{entity_id}/related")
async def get_entity_relations(entity_id: str):
    """获取实体相关的文档和关系"""
    try:
        return await knowledge_graph_service.get_entity_relations(entity_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 