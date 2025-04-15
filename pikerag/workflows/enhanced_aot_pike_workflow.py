"""
增强版的AoTPike工作流，集成miniRAG的异构图功能
"""

from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.knowledge_retrievers.enhanced_pike_retriever import EnhancedPikeRetriever
from pikerag.workflows.aot_pike_workflow import AoTPikeWorkflow


class EnhancedAoTPikeWorkflow(AoTPikeWorkflow):
    """
    增强版的AoTPike工作流，集成miniRAG的异构图功能
    """
    
    def _init_retriever(self) -> None:
        """初始化检索器"""
        super()._init_retriever()
        
        # 确保检索器是EnhancedPikeRetriever
        if not isinstance(self._retriever, EnhancedPikeRetriever):
            self.logger.warning("检索器不是EnhancedPikeRetriever，无法使用增强功能")
            return
            
        # 配置参数
        workflow_configs = self._yaml_config["workflow"].get("args", {})
        self._graph_search_depth = workflow_configs.get("graph_search_depth", 2)
        
    async def _answer_node_question(self, question: str, chosen_atom_infos: List[AtomRetrievalInfo]) -> Tuple[bool, Dict]:
        """
        增强的节点问题回答功能，使用融合检索
        
        Args:
            question: 问题
            chosen_atom_infos: 已选择的原子信息
            
        Returns:
            (成功标志, 回答结果)
        """
        if not isinstance(self._retriever, EnhancedPikeRetriever):
            return await super()._answer_node_question(question, chosen_atom_infos)
            
        # 使用融合检索
        retrieval_results = await self._retriever.aretrieve_with_fusion(
            query=question,
            retrieve_id=f"node_question_{hash(question)}",
            graph_search_depth=self._graph_search_depth
        )
        
        # 过滤已选择的信息
        filtered_results = self._filter_atom_candidates(retrieval_results, chosen_atom_infos)
        
        # 合并结果
        all_atom_infos = chosen_atom_infos + filtered_results
        
        # 使用LLM回答问题
        success, answer = await self._generate_answer(question, all_atom_infos)
        
        return success, answer
    
    async def process_document(self, document: str, doc_id: str = None) -> bool:
        """
        处理文档，抽取实体和关系，更新异构图
        
        Args:
            document: 文档内容
            doc_id: 文档ID
            
        Returns:
            是否成功处理
        """
        if not isinstance(self._retriever, EnhancedPikeRetriever):
            self.logger.warning("检索器不是EnhancedPikeRetriever，无法处理文档")
            return False
            
        return await self._retriever.process_document(document, doc_id)
    
    def _filter_atom_candidates(self, candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo]) -> List[AtomRetrievalInfo]:
        """
        过滤已选择的原子信息
        
        Args:
            candidates: 候选原子信息
            chosen_atom_infos: 已选择的原子信息
            
        Returns:
            过滤后的候选原子信息
        """
        if not chosen_atom_infos:
            return candidates
            
        # 过滤掉已选择的chunk
        chosen_chunk_ids = {info.source_chunk_id for info in chosen_atom_infos}
        filtered_candidates = []
        
        for candidate in candidates:
            if candidate.source_chunk_id not in chosen_chunk_ids:
                filtered_candidates.append(candidate)
                
        return filtered_candidates
    
    async def answer(self, qa, question_idx: int) -> Dict:
        """
        使用增强版的AoT-PIKE混合方式回答问题
        
        Args:
            qa: QA数据
            question_idx: 问题索引
            
        Returns:
            回答结果
        """
        # 检查是否有文档需要处理
        if hasattr(qa, "documents") and qa.documents:
            for doc_idx, doc in enumerate(qa.documents):
                await self.process_document(doc, f"doc_{doc_idx}")
        
        # 使用原有的回答方法
        return await super().answer(qa, question_idx) 