# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Any, Union
import asyncio
from xml.dom.minidom import Document
from langchain_chroma import Chroma
import numpy as np

from pikerag.knowledge_retrievers.chunk_atom_retriever import ChunkAtomRetriever, AtomRetrievalInfo
from pikerag.llm_client import BaseLLMClient
from pikerag.prompts import CommunicationProtocol
from pikerag.utils.logger import Logger


class AoTPikeRetriever(ChunkAtomRetriever):
    """结合AoT和PIKE-RAG的混合检索器，为DAG中的节点提供异构图检索能力"""

    name: str = "AoTPikeRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        """初始化AoTPikeRetriever"""
        super().__init__(retriever_config, log_dir, main_logger)
        
        # 加载AoT特定的配置
        aot_config = retriever_config.get("aot_config", {})
        self.markov_state_threshold = aot_config.get("markov_state_threshold", 0.8)
        
        # 动态原子查询生成相关
        self.enable_dynamic_atoms = aot_config.get("enable_dynamic_atoms", True)
        self.dynamic_atom_generation_protocol = None
        self.llm_client = None
        self.llm_config = {}
        
        # 异构图更新配置（消融实验选项）
        self.enable_graph_update = aot_config.get("enable_graph_update", True)
        self.max_dynamic_atoms_per_node = aot_config.get("max_dynamic_atoms_per_node", 5)
        
        # 并行处理配置
        self.enable_parallel = retriever_config.get("enable_parallel", False)
        self.max_parallel_tasks = retriever_config.get("max_parallel_tasks", 5)
    
    def set_llm_client(self, llm_client: BaseLLMClient, llm_config: Dict = None):
        """设置LLM客户端，用于动态生成原子查询"""
        self.llm_client = llm_client
        self.llm_config = llm_config or {}
    
    def set_dynamic_atom_generation_protocol(self, protocol: CommunicationProtocol):
        """设置动态原子查询生成协议"""
        self.dynamic_atom_generation_protocol = protocol
    
    async def aretrieve_for_dag_nodes_parallel(self, dag_nodes: List[Dict], current_state: str, 
                                             chosen_atom_infos: List[AtomRetrievalInfo] = None) -> Dict[str, List[AtomRetrievalInfo]]:
        """并行为多个DAG节点异步检索相关信息
        
        Args:
            dag_nodes: DAG中的节点信息列表
            current_state: 当前的马尔可夫状态
            chosen_atom_infos: 已选择的原子信息列表
            
        Returns:
            节点ID到检索到的原子信息列表的映射
        """
        if not self.enable_parallel or len(dag_nodes) <= 1:
            # 如果未启用并行或只有一个节点，使用顺序处理
            result = {}
            for node in dag_nodes:
                result[node["id"]] = await self.aretrieve_for_dag_node(node, current_state, chosen_atom_infos)
            return result
        
        self.logger.info(f"开始并行处理 {len(dag_nodes)} 个DAG节点的检索")
        
        async def process_node(node):
            node_id = node["id"]
            try:
                results = await self.aretrieve_for_dag_node(node, current_state, chosen_atom_infos)
                return node_id, results
            except Exception as e:
                self.logger.error(f"节点 {node_id} 检索失败: {e}")
                return node_id, []
        
        # 创建任务
        tasks = [process_node(node) for node in dag_nodes]
        
        # 以固定批次大小并行执行任务
        results = {}
        batch_size = min(self.max_parallel_tasks, len(tasks))
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            for node_id, node_results in batch_results:
                results[node_id] = node_results
        
        return results
    
    async def aretrieve_for_dag_node(self, dag_node: Dict, current_state: str, chosen_atom_infos: List[AtomRetrievalInfo] = None) -> List[AtomRetrievalInfo]:
        """异步为DAG中的节点检索相关信息
        
        Args:
            dag_node: DAG中的节点信息，包含id和question
            current_state: 当前的马尔可夫状态
            chosen_atom_infos: 已选择的原子信息列表
            
        Returns:
            检索到的原子信息列表
        """
        if chosen_atom_infos is None:
            chosen_atom_infos = []
            
        node_id = dag_node["id"]
        node_question = dag_node["question"]
        
        # 1. 使用节点问题在atom_store中检索
        self.logger.info(f"为节点 {node_id} 检索相关信息: {node_question}")
        atom_candidates = await self.aretrieve_atom_info_through_atom(
            queries=node_question,
            retrieve_id=f"dag_node_{node_id}"
        )
        
        # 过滤已选择的原子信息
        atom_candidates = self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
        
        # 2. 如果没有找到匹配的原子查询，尝试使用当前状态在atom_store中检索
        if len(atom_candidates) == 0:
            self.logger.info(f"节点 {node_id} 未找到直接匹配，尝试使用当前状态检索")
            atom_candidates = await self.aretrieve_atom_info_through_atom(
                queries=current_state,
                retrieve_id=f"dag_node_{node_id}_state"
            )
            atom_candidates = self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
        
        # 3. 如果仍未找到，尝试在chunk_store中检索
        if len(atom_candidates) == 0:
            self.logger.info(f"节点 {node_id} 在atom_store中未找到匹配，尝试在chunk_store中检索")
            atom_candidates = await self.aretrieve_atom_info_through_chunk(
                query=node_question,
                retrieve_id=f"dag_node_{node_id}_chunk"
            )
            atom_candidates = self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
        
        # 4. 如果启用了动态原子查询生成且仍未找到匹配，尝试生成新的原子查询
        if len(atom_candidates) == 0 and self.enable_dynamic_atoms and self.llm_client and self.dynamic_atom_generation_protocol:
            self.logger.info(f"节点 {node_id} 尝试动态生成原子查询")
            atom_candidates = await self._agenerate_and_retrieve_dynamic_atoms(
                node_question, current_state, chosen_atom_infos
            )
        
        # 根据与节点问题的相关性对结果排序
        if atom_candidates:
            atom_candidates = self._rank_by_relevance(node_question, atom_candidates)
            
        return atom_candidates
    
    def _filter_atom_candidates(self, candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo]) -> List[AtomRetrievalInfo]:
        """过滤已选择的原子信息"""
        if not chosen_atom_infos:
            return candidates
            
        # 过滤掉已选择的chunk
        chosen_chunk_ids = {info.source_chunk_id for info in chosen_atom_infos}
        filtered_candidates = []
        
        for candidate in candidates:
            if candidate.source_chunk_id not in chosen_chunk_ids:
                filtered_candidates.append(candidate)
            else:
                self.logger.debug(f"过滤掉已选择的chunk: {candidate.source_chunk_id}")
                
        return filtered_candidates
    
    def _rank_by_relevance(self, query: str, candidates: List[AtomRetrievalInfo]) -> List[AtomRetrievalInfo]:
        """根据与查询的相关性对候选进行排序"""
        query_embedding = self.embedding_func.embed_query(query)
        
        # 计算每个候选与查询的相似度
        for candidate in candidates:
            atom_embedding = candidate.atom_embedding
            similarity = self.similarity_func(query_embedding, atom_embedding)
            candidate.retrieval_score = similarity
            
        # 按相似度降序排序
        return sorted(candidates, key=lambda x: x.retrieval_score, reverse=True)
    
    async def aretrieve_atom_info_through_atom(
        self, queries: Union[List[str], str], retrieve_id: str="", **kwargs,
    ) -> List[AtomRetrievalInfo]:
        """异步通过原子查询检索相关信息"""
        # Decide which retrieve_k to use.
        if "retrieve_k" in kwargs:
            retrieve_k: int = kwargs["retrieve_k"]
        elif isinstance(queries, list) and len(queries) > 1:
            retrieve_k: int = self.atom_retrieve_k
        else:
            retrieve_k: int = self.retrieve_k

        # Wrap atom_queries into a list if only one element given.
        if isinstance(queries, str):
            queries = [queries]

        # Query `_atom_store` to get relevant atom information.
        query_atom_score_tuples: List[Tuple[str, Document, float]] = []
        for atom_query in queries:
            for atom_doc, score in await self._aget_doc_with_query(atom_query, self._atom_store, retrieve_k):
                query_atom_score_tuples.append((atom_query, atom_doc, score))

        # Wrap to predefined dataclass.
        return self._atom_info_tuple_to_class(query_atom_score_tuples)

    async def aretrieve_atom_info_through_chunk(self, query: str, retrieve_id: str="") -> List[AtomRetrievalInfo]:
        """异步通过chunk检索相关信息"""
        # Query `_chunk_store` to get relevant chunk information.
        chunk_info: List[Tuple[Document, float]] = await self._aget_doc_with_query(query, self._chunk_store, self.retrieve_k)

        # Wrap to predefined dataclass.
        return self._chunk_info_tuple_to_class(query=query, chunk_docs=[doc for doc, _ in chunk_info])

    async def _aget_doc_with_query(
        self, query: str, store: Chroma, retrieve_k: int=None, score_threshold: float=None,
    ) -> List[Tuple[Document, float]]:
        """异步使用查询从向量存储中获取文档"""
        if retrieve_k is None:
            retrieve_k = self.retrieve_k
        if score_threshold is None:
            score_threshold = self.retrieve_score_threshold

        # 注意：这里我们仍然使用同步方法，因为Chroma目前不支持异步API
        # 在未来Chroma支持异步API时可以更新此处
        infos: List[Tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query=query,
            k=retrieve_k,
            score_threshold=score_threshold,
        )

        filtered_docs = [(doc, score) for doc, score in infos if score >= score_threshold]
        sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)

        return sorted_docs

    async def _agenerate_and_retrieve_dynamic_atoms(
        self, node_question: str, current_state: str, chosen_atom_infos: List[AtomRetrievalInfo]
    ) -> List[AtomRetrievalInfo]:
        """异步生成新的原子查询并检索相关信息"""
        try:
            # 准备上下文
            context = ""
            if chosen_atom_infos:
                context_chunks = [info.source_chunk for info in chosen_atom_infos]
                context = "\n\n".join(context_chunks)
            
            # 生成原子查询
            messages = self.dynamic_atom_generation_protocol.process_input(
                node_question=node_question,
                current_state=current_state,
                context=context
            )
            
            # 异步生成内容
            response = await self.llm_client.agenerate_content_with_messages(messages, **self.llm_config)
            success, new_atoms = self.dynamic_atom_generation_protocol.parse_output(response)
            
            if not success or not new_atoms:
                self.logger.warning("动态原子查询生成失败")
                return []
                
            self.logger.info(f"生成了 {len(new_atoms)} 个新的原子查询")
            
            # 限制生成的原子查询数量
            if len(new_atoms) > self.max_dynamic_atoms_per_node:
                self.logger.info(f"限制动态原子查询数量从 {len(new_atoms)} 到 {self.max_dynamic_atoms_per_node}")
                new_atoms = new_atoms[:self.max_dynamic_atoms_per_node]
            
            # 使用新生成的原子查询检索
            atom_candidates = await self.aretrieve_atom_info_through_atom(
                queries=new_atoms,
                retrieve_id=f"dynamic_atoms_{hash(node_question)}"
            )
            
            # 如果启用了异构图更新，则更新图
            if self.enable_graph_update:
                await self.aupdate_graph_with_new_atoms(new_atoms, node_question)
            
            return self._filter_atom_candidates(atom_candidates, chosen_atom_infos)
            
        except Exception as e:
            self.logger.error(f"动态原子查询生成错误: {e}")
            return []

    async def aupdate_graph_with_new_atoms(self, atoms: List[str], source_text: str, source_id: str = None):
        """异步使用新生成的原子查询更新异构图"""
        if not atoms or not self.enable_graph_update:
            return
            
        # 生成源ID
        if source_id is None:
            source_id = f"dynamic_chunk_{hash(source_text)}"
            
        # 检查chunk是否已存在
        chunk_exists = False
        try:
            chunk_docs = self._chunk_store.get(ids=[source_id])
            if chunk_docs and len(chunk_docs["documents"]) > 0:
                chunk_exists = True
        except Exception as e:
            self.logger.warning(f"检查chunk存在性时出错: {e}")
            
        # 如果chunk不存在，创建新的chunk
        if not chunk_exists:
            self._chunk_store.add_texts(
                texts=[source_text],
                ids=[source_id],
                metadatas=[{"dynamic": True, "atom_questions_str": "\n".join(atoms)}]
            )
            self.logger.info(f"创建了新的chunk: {source_id}")
            
        # 为每个原子查询创建嵌入并添加到atom_store
        atom_ids = []
        atom_texts = []
        atom_metadatas = []
        
        for atom in atoms:
            atom_id = f"dynamic_atom_{hash(atom)}"
            atom_ids.append(atom_id)
            atom_texts.append(atom)
            atom_metadatas.append({
                "source_chunk_id": source_id,
                "dynamic": True
            })
            
        # 添加到atom_store
        self._atom_store.add_texts(
            texts=atom_texts,
            ids=atom_ids,
            metadatas=atom_metadatas
        )
        
        self.logger.info(f"添加了 {len(atoms)} 个新的原子查询到异构图")
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取检索器的统计信息"""
        stats = super().get_statistics() if hasattr(super(), "get_statistics") else {}
        
        # 统计异构图中的动态原子查询信息
        try:
            dynamic_atoms_count = 0
            dynamic_chunks_count = 0
            
            # 统计动态原子查询
            atom_metadatas = self._atom_store.get()["metadatas"]
            for metadata in atom_metadatas:
                if metadata and metadata.get("dynamic", False):
                    dynamic_atoms_count += 1
            
            # 统计动态chunks
            chunk_metadatas = self._chunk_store.get()["metadatas"]
            for metadata in chunk_metadatas:
                if metadata and metadata.get("dynamic", False):
                    dynamic_chunks_count += 1
                    
            stats.update({
                "dynamic_atoms_count": dynamic_atoms_count,
                "dynamic_chunks_count": dynamic_chunks_count
            })
        except Exception as e:
            self.logger.error(f"获取统计信息时出错: {e}")
            
        return stats 