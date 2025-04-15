# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from tqdm import tqdm


def load_hotpotqa_documents(data_path: str) -> Tuple[List[str], List[Document]]:
    """
    加载HotpotQA数据集中的文档
    
    Args:
        data_path: HotpotQA数据集的路径
        
    Returns:
        文档ID列表和文档对象列表的元组
    """
    print(f"从 {data_path} 加载HotpotQA文档...")
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    doc_ids = []
    documents = []
    
    # 处理每个问题
    for item in tqdm(data, desc="处理文档"):
        # 处理每个支持事实
        
        for title, sentences in item['context']:
            # 为每个段落创建一个文档
            doc_id = f"doc_{title.replace(' ', '_')}"
            content = " ".join(sentences)
            
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "id": doc_id,
                            "title": title
                        }
                    )
                )
    
    print(f"加载了 {len(documents)} 个文档")
    return doc_ids, documents


def load_hotpotqa_atoms(data_path: str, atom_tag: str = "question") -> Tuple[None, List[Document]]:
    """
    从HotpotQA数据集中提取原子查询
    
    Args:
        data_path: HotpotQA数据集的路径
        atom_tag: 原子查询的标签
        
    Returns:
        None和原子查询文档列表的元组
    """
    print(f"从 {data_path} 提取原子查询...")
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    atom_docs = []
    
    # 处理每个问题
    for item in data:
        question = item['question']
        
        # 将问题作为原子查询
        atom_docs.append(
            Document(
                page_content=question,
                metadata={
                    "source_id": item['_id'],
                    "type": atom_tag
                }
            )
        )
        
        # 从问题中提取实体作为原子查询
        entities = extract_entities_from_question(question)
        for entity in entities:
            atom_docs.append(
                Document(
                    page_content=f"关于{entity}的信息是什么?",
                    metadata={
                        "source_id": item['_id'],
                        "type": "entity_query",
                        "entity": entity
                    }
                )
            )
    
    print(f"提取了 {len(atom_docs)} 个原子查询")
    return None, atom_docs


def extract_entities_from_question(question: str) -> List[str]:
    """
    从问题中提取实体
    
    Args:
        question: 问题文本
        
    Returns:
        提取的实体列表
    """
    # 简单的启发式方法：提取大写开头的连续词组
    import re
    
    # 移除问号和其他标点
    question = re.sub(r'[^\w\s]', '', question)
    
    # 分词
    words = question.split()
    
    entities = []
    current_entity = []
    
    for word in words:
        # 如果是大写开头的词，可能是实体的一部分
        if word and word[0].isupper():
            current_entity.append(word)
        elif current_entity:
            # 当前词不是大写开头，但之前有积累的实体
            if len(current_entity) > 0:
                entities.append(" ".join(current_entity))
                current_entity = []
    
    # 处理最后可能剩余的实体
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities 