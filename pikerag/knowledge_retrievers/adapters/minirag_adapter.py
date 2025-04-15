"""
miniRAG适配器模块 - 将miniRAG的异构图构建功能引入到pikeRAG中
"""

import asyncio
import re
from typing import Dict, List, Tuple, Any, Union, Optional
import importlib
from dataclasses import dataclass

# 导入pikeRAG相关模块
from pikerag.utils.logger import Logger
from pikerag.knowledge_retrievers.chunk_atom_retriever import ChunkAtomRetriever, AtomRetrievalInfo
from pikerag.llm_client import BaseLLMClient

# 尝试导入miniRAG模块
try:
    from minirag.operate import (
        extract_entities,
        _handle_single_entity_extraction,
        _handle_single_relationship_extraction,
        _merge_nodes_then_upsert,
        _merge_edges_then_upsert,
        split_string_by_multi_markers,
        clean_str,
    )
    from minirag.prompt import PROMPTS as MINIRAG_PROMPTS
    from minirag.base import BaseGraphStorage
    MINIRAG_AVAILABLE = True
except ImportError:
    MINIRAG_AVAILABLE = False
    # 创建占位函数，以便在miniRAG不可用时不会出错
    async def extract_entities(*args, **kwargs):
        return None
    
    async def _handle_single_entity_extraction(*args, **kwargs):
        return None
    
    async def _handle_single_relationship_extraction(*args, **kwargs):
        return None
    
    async def _merge_nodes_then_upsert(*args, **kwargs):
        return None
    
    async def _merge_edges_then_upsert(*args, **kwargs):
        return None
    
    def split_string_by_multi_markers(*args, **kwargs):
        return []
    
    def clean_str(s):
        return s
    
    MINIRAG_PROMPTS = {}
    
    @dataclass
    class BaseGraphStorage:
        pass


class MiniRAGAdapter:
    """
    适配器类，用于将miniRAG的异构图构建功能引入到pikeRAG中
    """
    
    def __init__(self, retriever: ChunkAtomRetriever, graph_storage_config: Dict = None):
        """
        初始化适配器
        
        Args:
            retriever: pikeRAG的检索器实例
            graph_storage_config: 图存储配置
        """
        self.retriever = retriever
        self.logger = retriever.logger
        self.graph_storage = None
        self.entity_vdb = None
        self.entity_name_vdb = None
        self.relationships_vdb = None
        
        # 检查miniRAG是否可用
        if not MINIRAG_AVAILABLE:
            self.logger.warning("miniRAG模块不可用，异构图构建功能将不可用")
            return
            
        # 初始化图存储
        if graph_storage_config:
            self._init_graph_storage(graph_storage_config)
    
    def _init_graph_storage(self, config: Dict):
        """
        初始化图存储
        
        Args:
            config: 图存储配置
        """
        if not MINIRAG_AVAILABLE:
            return
            
        try:
            # 动态导入图存储类
            storage_type = config.get("storage_type", "NetworkXStorage")
            module_path = config.get("module_path", "minirag.kg.networkx_impl")
            
            module = importlib.import_module(module_path)
            storage_class = getattr(module, storage_type)
            
            # 创建图存储实例
            self.graph_storage = storage_class(
                namespace=config.get("namespace", "pikerag"),
                global_config=config.get("global_config", {}),
                embedding_func=self.retriever.embedding_func
            )
            
            self.logger.info(f"成功初始化图存储: {storage_type}")
            
            # 初始化向量存储
            # 这里可以根据需要初始化实体和关系的向量存储
            # 暂时使用None，后续可以扩展
            
        except Exception as e:
            self.logger.error(f"初始化图存储失败: {e}")
    
    async def extract_entities_from_text(self, text: str, chunk_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        从文本中抽取实体和关系
        
        Args:
            text: 输入文本
            chunk_id: 文本块ID
            
        Returns:
            实体列表和关系列表
        """
        if not MINIRAG_AVAILABLE or not hasattr(self.retriever, "llm_client") or not self.retriever.llm_client:
            return [], []
            
        # 使用miniRAG的实体抽取提示模板
        entity_extract_prompt = MINIRAG_PROMPTS.get("entity_extraction", "")
        if not entity_extract_prompt:
            self.logger.warning("实体抽取提示模板不可用")
            return [], []
            
        context_base = dict(
            tuple_delimiter=MINIRAG_PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|>"),
            record_delimiter=MINIRAG_PROMPTS.get("DEFAULT_RECORD_DELIMITER", "##"),
            completion_delimiter=MINIRAG_PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|COMPLETE|>"),
            entity_types=",".join(MINIRAG_PROMPTS.get("DEFAULT_ENTITY_TYPES", ["organization", "person", "location", "event"])),
        )
        
        # 生成提示
        prompt = entity_extract_prompt.format(**context_base, input_text=text)
        
        # 调用LLM抽取实体和关系
        try:
            response = await self.retriever.llm_client.agenerate_content_with_messages([{"role": "user", "content": prompt}])
            response_text = response.get("content", "")
            
            # 解析LLM响应
            records = split_string_by_multi_markers(
                response_text,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            
            # 处理实体和关系
            entities = []
            relations = []
            
            for record in records:
                record_match = re.search(r"\((.*)\)", record)
                if not record_match:
                    continue
                    
                record_text = record_match.group(1)
                attributes = split_string_by_multi_markers(record_text, [context_base["tuple_delimiter"]])
                
                if len(attributes) >= 4 and attributes[0] == '"entity"':
                    entity_name = clean_str(attributes[1].upper())
                    if not entity_name.strip():
                        continue
                    entity_type = clean_str(attributes[2].upper())
                    entity_description = clean_str(attributes[3])
                    
                    entities.append({
                        "entity_name": entity_name,
                        "entity_type": entity_type, 
                        "description": entity_description,
                        "source_id": chunk_id
                    })
                    
                elif len(attributes) >= 6 and attributes[0] == '"relationship"':
                    src_entity = clean_str(attributes[1].upper())
                    tgt_entity = clean_str(attributes[2].upper())
                    relation_desc = clean_str(attributes[3])
                    keywords = clean_str(attributes[4])
                    try:
                        weight = float(attributes[5])
                    except:
                        weight = 5.0
                    
                    relations.append({
                        "src_id": src_entity,
                        "tgt_id": tgt_entity,
                        "description": relation_desc,
                        "keywords": keywords,
                        "weight": weight,
                        "source_id": chunk_id
                    })
            
            self.logger.info(f"从文本中抽取了 {len(entities)} 个实体和 {len(relations)} 个关系")
            return entities, relations
            
        except Exception as e:
            self.logger.error(f"实体抽取失败: {e}")
            return [], []
    
    async def update_graph(self, entities: List[Dict], relations: List[Dict]) -> bool:
        """
        更新图结构
        
        Args:
            entities: 实体列表
            relations: 关系列表
            
        Returns:
            是否成功更新
        """
        if not MINIRAG_AVAILABLE or not self.graph_storage:
            return False
            
        try:
            # 更新实体节点
            for entity in entities:
                await self.graph_storage.upsert_node(
                    entity["entity_name"],
                    node_data={
                        "source_id": entity["source_id"],
                        "description": entity["description"],
                        "entity_type": f'"{entity["entity_type"]}"',
                    }
                )
                
            # 更新关系边
            for relation in relations:
                await self.graph_storage.upsert_edge(
                    relation["src_id"],
                    relation["tgt_id"],
                    edge_data={
                        "weight": relation["weight"],
                        "description": relation["description"],
                        "keywords": relation["keywords"],
                        "source_id": relation["source_id"],
                    }
                )
                
            self.logger.info(f"成功更新图结构: {len(entities)} 个实体和 {len(relations)} 个关系")
            return True
            
        except Exception as e:
            self.logger.error(f"更新图结构失败: {e}")
            return False
    
    async def generate_atoms_from_entities(self, entities: List[Dict], relations: List[Dict], text: str, source_id: str) -> List[str]:
        """
        从实体和关系生成原子查询
        
        Args:
            entities: 实体列表
            relations: 关系列表
            text: 原始文本
            source_id: 源文本ID
            
        Returns:
            生成的原子查询列表
        """
        atoms = []
        
        # 从实体生成查询
        for entity in entities:
            atoms.append(f"关于{entity['entity_name']}的信息是什么？")
            atoms.append(f"{entity['entity_name']}是什么类型的{entity['entity_type']}？")
            
        # 从关系生成查询
        for relation in relations:
            atoms.append(f"{relation['src_id']}和{relation['tgt_id']}之间有什么关系？")
            atoms.append(f"{relation['src_id']}如何影响{relation['tgt_id']}？")
        
        # 限制原子查询数量
        max_atoms = getattr(self.retriever, "max_dynamic_atoms_per_node", 10)
        if len(atoms) > max_atoms:
            atoms = atoms[:max_atoms]
        
        return atoms
    
    async def process_text(self, text: str, chunk_id: str = None) -> bool:
        """
        处理文本，抽取实体和关系，更新图结构，生成原子查询
        
        Args:
            text: 输入文本
            chunk_id: 文本块ID
            
        Returns:
            是否成功处理
        """
        if not chunk_id:
            chunk_id = f"chunk_{hash(text)}"
            
        # 1. 抽取实体和关系
        entities, relations = await self.extract_entities_from_text(text, chunk_id)
        if not entities and not relations:
            return False
            
        # 2. 更新图结构
        if self.graph_storage:
            await self.update_graph(entities, relations)
            
        # 3. 生成原子查询并更新pikeRAG的异构图
        atoms = await self.generate_atoms_from_entities(entities, relations, text, chunk_id)
        if atoms and hasattr(self.retriever, "aupdate_graph_with_new_atoms"):
            await self.retriever.aupdate_graph_with_new_atoms(atoms, text, chunk_id)
            
        return True
    
    async def query_graph(self, query: str, top_k: int = 5, depth: int = 2) -> List[Dict]:
        """
        查询图结构
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            depth: 图遍历深度
            
        Returns:
            查询结果列表
        """
        if not MINIRAG_AVAILABLE or not self.graph_storage:
            return []
            
        try:
            # 这里实现图查询逻辑
            # 由于miniRAG的图查询逻辑较为复杂，这里简化处理
            # 实际应用中可以根据需要扩展
            
            # 1. 获取所有实体类型
            types, _ = await self.graph_storage.get_types()
            
            # 2. 获取所有实体
            entities = await self.graph_storage.get_node_from_types(types)
            
            # 3. 根据查询文本过滤实体
            # 这里使用简单的字符串匹配，实际应用中可以使用向量相似度
            filtered_entities = []
            for entity in entities:
                entity_name = entity.get("entity_name", "")
                description = entity.get("description", "")
                if query.lower() in entity_name.lower() or query.lower() in description.lower():
                    filtered_entities.append(entity)
            
            # 4. 获取实体的邻居
            results = []
            for entity in filtered_entities[:top_k]:
                entity_id = entity.get("entity_name", "")
                neighbors = await self.graph_storage.get_neighbors_within_k_hops(entity_id, depth)
                
                # 获取邻居实体的详细信息
                neighbor_entities = set()
                for edge in neighbors:
                    neighbor_entities.add(edge[0])
                    neighbor_entities.add(edge[1])
                
                neighbor_info = []
                for neighbor_id in neighbor_entities:
                    neighbor = await self.graph_storage.get_node(neighbor_id)
                    if neighbor:
                        neighbor_info.append({
                            "entity_name": neighbor_id,
                            "description": neighbor.get("description", ""),
                            "entity_type": neighbor.get("entity_type", "")
                        })
                
                results.append({
                    "entity": entity,
                    "neighbors": neighbor_info,
                    "edges": neighbors
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"图查询失败: {e}")
            return []
    
    async def convert_graph_results_to_atom_info(self, graph_results: List[Dict], query: str) -> List[AtomRetrievalInfo]:
        """
        将图查询结果转换为AtomRetrievalInfo格式
        
        Args:
            graph_results: 图查询结果
            query: 原始查询
            
        Returns:
            AtomRetrievalInfo列表
        """
        if not graph_results:
            return []
            
        results = []
        for result in graph_results:
            entity = result.get("entity", {})
            neighbors = result.get("neighbors", [])
            
            entity_name = entity.get("entity_name", "")
            description = entity.get("description", "")
            source_id = entity.get("source_id", "")
            
            # 创建一个虚拟原子查询
            atom_query = f"关于{entity_name}的信息是什么？"
            
            # 构建源文本
            source_chunk = f"{entity_name}: {description}\n\n相关实体:\n"
            for neighbor in neighbors:
                source_chunk += f"- {neighbor.get('entity_name', '')}: {neighbor.get('description', '')}\n"
            
            # 计算嵌入
            atom_embedding = self.retriever.embedding_func.embed_query(entity_name + " " + description)
            
            # 创建检索信息对象
            retrieval_info = AtomRetrievalInfo(
                atom_query=atom_query,
                atom=f"{entity_name}: {description}",
                source_chunk_title=entity_name,
                source_chunk=source_chunk,
                source_chunk_id=source_id,
                retrieval_score=0.8,  # 默认分数
                atom_embedding=atom_embedding,
            )
            
            results.append(retrieval_info)
        
        return results 