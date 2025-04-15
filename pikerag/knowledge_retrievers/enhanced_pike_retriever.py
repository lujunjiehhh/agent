"""
增强版的PIKE检索器，集成miniRAG的异构图功能
"""

from typing import Dict, List, Tuple, Any, Union, Optional
import asyncio

from pikerag.knowledge_retrievers.aot_pike_retriever import AoTPikeRetriever
from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.utils.logger import Logger
from pikerag.llm_client import BaseLLMClient

# 导入适配器
from pikerag.knowledge_retrievers.adapters.minirag_adapter import MiniRAGAdapter


class EnhancedPikeRetriever(AoTPikeRetriever):
    """
    增强版的PIKE检索器，集成miniRAG的异构图功能
    """
    
    name: str = "EnhancedPikeRetriever"
    
    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        """
        初始化增强版PIKE检索器
        
        Args:
            retriever_config: 检索器配置
            log_dir: 日志目录
            main_logger: 主日志记录器
        """
        super().__init__(retriever_config, log_dir, main_logger)
        
        # 加载miniRAG相关配置
        minirag_config = retriever_config.get("minirag_config", {})
        self.enable_minirag = minirag_config.get("enable_minirag", True)
        self.enable_entity_extraction = minirag_config.get("enable_entity_extraction", True)
        self.enable_graph_query = minirag_config.get("enable_graph_query", True)
        self.fusion_mode = minirag_config.get("fusion_mode", "hybrid")  # hybrid, pike_only, graph_only
        
        # 初始化miniRAG适配器
        self.minirag_adapter = None
        if self.enable_minirag:
            graph_storage_config = minirag_config.get("graph_storage", {})
            self.minirag_adapter = MiniRAGAdapter(self, graph_storage_config)
            self.logger.info("初始化miniRAG适配器成功")
    
    async def process_document(self, document: str, doc_id: str = None) -> bool:
        """
        处理文档，抽取实体和关系，更新异构图
        
        Args:
            document: 文档内容
            doc_id: 文档ID
            
        Returns:
            是否成功处理
        """
        if not self.enable_minirag or not self.minirag_adapter:
            return False
            
        if doc_id is None:
            doc_id = f"doc_{hash(document)}"
            
        # 分块处理文档
        # 这里使用简单的分块方法，实际应用中可以使用更复杂的分块策略
        max_chunk_size = 1000
        chunks = [document[i:i+max_chunk_size] for i in range(0, len(document), max_chunk_size)]
        
        results = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # 添加到chunk_store
            self._chunk_store.add_texts(
                texts=[chunk],
                ids=[chunk_id],
                metadatas=[{"doc_id": doc_id}]
            )
            
            # 使用miniRAG适配器处理文本
            result = await self.minirag_adapter.process_text(chunk, chunk_id)
            results.append(result)
            
        return any(results)
    
    async def aretrieve_with_fusion(
        self, query: str, retrieve_id: str = "", graph_search_depth: int = 2
    ) -> List[AtomRetrievalInfo]:
        """
        融合PIKE和miniRAG图查询的检索方法
        
        Args:
            query: 查询文本
            retrieve_id: 检索ID
            graph_search_depth: 图搜索深度
            
        Returns:
            检索结果列表
        """
        results = []
        
        # 使用融合模式决定检索策略
        if self.fusion_mode in ["pike_only", "hybrid"]:
            # 通过PIKE检索
            pike_results = await self.aretrieve_atom_info_through_atom(
                queries=query, retrieve_id=f"{retrieve_id}_pike"
            )
            results.extend(pike_results)
        
        if self.fusion_mode in ["graph_only", "hybrid"] and self.enable_graph_query and self.minirag_adapter:
            # 通过图检索
            graph_results = await self.minirag_adapter.query_graph(
                query, top_k=self.retrieve_k, depth=graph_search_depth
            )
            
            # 转换为AtomRetrievalInfo格式
            graph_atom_results = await self.minirag_adapter.convert_graph_results_to_atom_info(graph_results, query)
            
            # 合并结果
            results.extend(graph_atom_results)
        
        # 对结果进行排序和去重
        unique_results = {}
        for result in results:
            if result.source_chunk_id not in unique_results:
                unique_results[result.source_chunk_id] = result
            elif result.retrieval_score > unique_results[result.source_chunk_id].retrieval_score:
                unique_results[result.source_chunk_id] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x.retrieval_score, reverse=True)
        
        # 限制结果数量
        return sorted_results[:self.retrieve_k]
    
    async def aretrieve_for_dag_node(self, dag_node: Dict, current_state: str, chosen_atom_infos: List[AtomRetrievalInfo] = None) -> List[AtomRetrievalInfo]:
        """
        为DAG节点检索信息，增强版本使用融合检索
        
        Args:
            dag_node: DAG节点
            current_state: 当前状态
            chosen_atom_infos: 已选择的原子信息
            
        Returns:
            检索结果列表
        """
        if not self.enable_minirag or not self.minirag_adapter:
            return await super().aretrieve_for_dag_node(dag_node, current_state, chosen_atom_infos)
            
        node_id = dag_node["id"]
        node_question = dag_node["question"]
        
        # 使用融合检索
        atom_candidates = await self.aretrieve_with_fusion(
            query=node_question,
            retrieve_id=f"dag_node_{node_id}",
            graph_search_depth=2
        )
        
        # 过滤已选择的原子信息
        atom_candidates = self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
        
        # 如果没有找到匹配的原子查询，尝试使用当前状态检索
        if len(atom_candidates) == 0:
            self.logger.info(f"节点 {node_id} 未找到直接匹配，尝试使用当前状态检索")
            atom_candidates = await self.aretrieve_with_fusion(
                query=current_state,
                retrieve_id=f"dag_node_{node_id}_state",
                graph_search_depth=2
            )
            atom_candidates = self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
        
        # 如果仍未找到，尝试在chunk_store中检索
        if len(atom_candidates) == 0:
            self.logger.info(f"节点 {node_id} 在融合检索中未找到匹配，尝试在chunk_store中检索")
            atom_candidates = await self.aretrieve_atom_info_through_chunk(
                query=node_question,
                retrieve_id=f"dag_node_{node_id}_chunk"
            )
            atom_candidates = self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
        
        # 如果启用了动态原子查询生成且仍未找到匹配，尝试生成新的原子查询
        if len(atom_candidates) == 0 and self.enable_dynamic_atoms and self.llm_client and self.dynamic_atom_generation_protocol:
            self.logger.info(f"节点 {node_id} 尝试动态生成原子查询")
            atom_candidates = await self._agenerate_and_retrieve_dynamic_atoms(
                node_question, current_state, chosen_atom_infos
            )
        
        # 根据与节点问题的相关性对结果排序
        if atom_candidates:
            atom_candidates = self._rank_by_relevance(node_question, atom_candidates)
            
        return atom_candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索器的统计信息
        
        Returns:
            统计信息字典
        """
        stats = super().get_statistics()
        
        # 添加miniRAG相关统计信息
        if self.enable_minirag and self.minirag_adapter and self.minirag_adapter.graph_storage:
            try:
                # 获取图统计信息
                types, _ = asyncio.run(self.minirag_adapter.graph_storage.get_types())
                entities = asyncio.run(self.minirag_adapter.graph_storage.get_node_from_types(types))
                
                stats.update({
                    "graph_entity_count": len(entities),
                    "graph_entity_types": len(types),
                    "minirag_enabled": self.enable_minirag,
                    "entity_extraction_enabled": self.enable_entity_extraction,
                    "graph_query_enabled": self.enable_graph_query,
                    "fusion_mode": self.fusion_mode
                })
            except Exception as e:
                self.logger.error(f"获取图统计信息失败: {e}")
        
        return stats 