from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    MD = "md"

class Document(BaseModel):
    id: str
    filename: str
    file_type: FileType
    created_at: datetime
    processed_at: Optional[datetime] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: FileType
    created_at: datetime
    processed_at: Optional[datetime] = None

class QuestionRequest(BaseModel):
    question: str

class ReasoningStep(BaseModel):
    step_id: str
    description: str
    type: str  # "decompose", "retrieve", "select", "generate"
    details: Optional[Dict[str, Any]] = None
    parent_step_id: Optional[str] = None

class Reference(BaseModel):
    doc_id: str
    content: str
    relevance_score: float
    position: Optional[Dict[str, int]] = None

class QuestionResponse(BaseModel):
    answer: str
    reasoning_steps: List[ReasoningStep]
    references: List[Reference]

class Entity(BaseModel):
    id: str
    name: str
    type: str
    properties: Optional[Dict[str, Any]] = None

class Relation(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: Optional[Dict[str, Any]] = None

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class EntityRelation(BaseModel):
    entity: Entity
    related_documents: List[DocumentResponse]
    relations: List[Relation] 