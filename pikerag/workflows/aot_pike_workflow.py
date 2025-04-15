# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Any, Optional
import asyncio
import numpy as np
import os
import jsonlines
from tqdm import tqdm

from pikerag.knowledge_retrievers.aot_pike_retriever import AoTPikeRetriever
from pikerag.knowledge_retrievers.chunk_atom_retriever import ChunkAtomRetriever, AtomRetrievalInfo
from pikerag.prompts.decomposition.aot_pike import (
    aot_dag_decomposition_protocol,
    aot_contract_protocol,
    aot_node_answering_protocol,
    aot_answerability_check_protocol,
    dynamic_atom_generation_protocol,
    ensemble_protocol,
    retry_with_backoff
)
from pikerag.utils.config_loader import load_protocol
from pikerag.utils.logger import Logger
from pikerag.workflows.common import BaseQaData
from pikerag.workflows.qa import QaWorkflow


class AoTPikeWorkflow(QaWorkflow):
    """结合AoT的马尔可夫推理特性与PIKE-RAG的异构图检索机制的工作流"""

    def __init__(self, yaml_config: Dict) -> None:
        """初始化AoTPikeWorkflow"""
        super().__init__(yaml_config)

        # 加载工作流配置
        workflow_configs: dict = self._yaml_config["workflow"].get("args", {})
        self._max_iterations: int = workflow_configs.get("max_iterations", 5)
        self._max_dag_depth: int = workflow_configs.get("max_dag_depth", 3)
        
        # 记忆机制配置（消融实验选项）
        self._enable_memory: bool = workflow_configs.get("enable_memory", False)
        self._memory_buffer_size: int = workflow_configs.get("memory_buffer_size", 3)
        self._memory_weight_decay: float = workflow_configs.get("memory_weight_decay", 0.8)
        self._memory_buffer: List[str] = []
        
        # 集成决策机制配置（消融实验选项）
        self._enable_ensemble: bool = workflow_configs.get("enable_ensemble", True)
        self._ensemble_methods: List[str] = workflow_configs.get("ensemble_methods", ["direct", "decompose", "contract"])
        
        # 重试机制配置（消融实验选项）
        self._enable_retry: bool = workflow_configs.get("enable_retry", True)
        self._max_retries: int = workflow_configs.get("max_retries", 3)
        self._retry_delay: float = workflow_configs.get("retry_delay", 1.0)
        
        # 并行处理配置（消融实验选项）
        self._enable_parallel: bool = workflow_configs.get("enable_parallel", True)
        self._max_parallel_nodes: int = workflow_configs.get("max_parallel_nodes", 5)
        
        # 插件模式配置（消融实验选项）
        self._enable_plugin_mode: bool = workflow_configs.get("enable_plugin_mode", False)
        self._plugin_samples: int = workflow_configs.get("plugin_samples", 3)
        
        # 初始化协议
        self._init_aot_protocols()
        
        # 日志记录器
        self._aot_logger = Logger("aot_pike", dump_folder=self._yaml_config["log_dir"])

    def _init_aot_protocols(self) -> None:
        """初始化AoT相关的协议"""
        # 使用默认协议
        self._dag_decompose_protocol = aot_dag_decomposition_protocol
        self._contract_protocol = aot_contract_protocol
        self._node_answering_protocol = aot_node_answering_protocol
        self._answerability_check_protocol = aot_answerability_check_protocol
        self._ensemble_protocol = ensemble_protocol
        
        # 如果配置中有自定义协议，则加载
        if "dag_decompose_protocol" in self._yaml_config:
            config = self._yaml_config["dag_decompose_protocol"]
            self._dag_decompose_protocol = load_protocol(
                module_path=config["module_path"],
                protocol_name=config["protocol_name"],
                partial_values=config.get("template_partial", {})
            )
            
        if "contract_protocol" in self._yaml_config:
            config = self._yaml_config["contract_protocol"]
            self._contract_protocol = load_protocol(
                module_path=config["module_path"],
                protocol_name=config["protocol_name"],
                partial_values=config.get("template_partial", {})
            )
            
        if "node_answering_protocol" in self._yaml_config:
            config = self._yaml_config["node_answering_protocol"]
            self._node_answering_protocol = load_protocol(
                module_path=config["module_path"],
                protocol_name=config["protocol_name"],
                partial_values=config.get("template_partial", {})
            )
            
        if "answerability_check_protocol" in self._yaml_config:
            config = self._yaml_config["answerability_check_protocol"]
            self._answerability_check_protocol = load_protocol(
                module_path=config["module_path"],
                protocol_name=config["protocol_name"],
                partial_values=config.get("template_partial", {})
            )
            
        if "ensemble_protocol" in self._yaml_config:
            config = self._yaml_config["ensemble_protocol"]
            self._ensemble_protocol = load_protocol(
                module_path=config["module_path"],
                protocol_name=config["protocol_name"],
                partial_values=config.get("template_partial", {})
            )

    def _init_retriever(self) -> None:
        """初始化检索器"""
        super()._init_retriever()
        
        # 确保检索器是AoTPikeRetriever或ChunkAtomRetriever
        if not isinstance(self._retriever, (AoTPikeRetriever, ChunkAtomRetriever)):
            raise TypeError(f"检索器类型必须是AoTPikeRetriever或ChunkAtomRetriever，而不是{type(self._retriever)}")
        
        # 如果是AoTPikeRetriever，设置LLM客户端和动态原子查询生成协议
        if isinstance(self._retriever, AoTPikeRetriever):
            self._retriever.set_llm_client(self._client, self.llm_config)
            self._retriever.set_dynamic_atom_generation_protocol(dynamic_atom_generation_protocol)
            
            # 设置并行处理配置
            self._retriever.enable_parallel = self._enable_parallel
            self._retriever.max_parallel_tasks = self._max_parallel_nodes

    @retry_with_backoff()
    async def _decompose_to_dag(self, question: str, chosen_atom_infos: List[AtomRetrievalInfo]) -> Tuple[bool, str, Dict[str, Any]]:
        """将问题分解为依赖型有向无环图(DAG)"""
        self._aot_logger.info(f"将问题分解为DAG: {question}")
        
        messages = self._dag_decompose_protocol.process_input(
            content=question,
            chosen_atom_infos=chosen_atom_infos
        )
        
        response = await self._client.agenerate_content_with_messages(messages, **self.llm_config)
        success, thinking, dag = self._dag_decompose_protocol.parse_output(response)
        
        if not success:
            self._aot_logger.warning("DAG分解失败")
            return False, thinking, {}
            
        self._aot_logger.info(f"DAG分解成功，生成了 {len(dag.get('nodes', []))} 个节点")
        return True, thinking, dag

    @retry_with_backoff()
    async def _check_if_answerable(self, current_state: str, chosen_atom_infos: List[AtomRetrievalInfo]) -> Tuple[bool, str]:
        """检查当前状态是否可以直接回答"""
        self._aot_logger.info(f"检查当前状态是否可以直接回答: {current_state}")
        
        messages = self._answerability_check_protocol.process_input(
            current_state=current_state,
            chosen_atom_infos=chosen_atom_infos
        )
        
        response = await self._client.agenerate_content_with_messages(messages, **self.llm_config)
        success, is_answerable, reason = self._answerability_check_protocol.parse_output(response)
        
        if not success:
            self._aot_logger.warning("可解答性检查失败")
            return False, "检查失败"
            
        self._aot_logger.info(f"可解答性检查结果: {is_answerable}, 原因: {reason}")
        return is_answerable, reason

    @retry_with_backoff()
    async def _answer_node_question(self, node_question: str, chosen_atom_infos: List[AtomRetrievalInfo]) -> Tuple[bool, Dict[str, Any]]:
        """回答节点问题"""
        self._aot_logger.info(f"回答节点问题: {node_question}")
        
        messages = self._node_answering_protocol.process_input(
            node_question=node_question,
            chosen_atom_infos=chosen_atom_infos
        )
        
        response = await self._client.agenerate_content_with_messages(messages, **self.llm_config)
        success, result = self._node_answering_protocol.parse_output(response)
        
        if not success:
            self._aot_logger.warning("节点问题回答失败")
            return False, {}
            
        self._aot_logger.info(f"节点问题回答成功: {result.get('answer', '')}")
        return True, result

    @retry_with_backoff()
    async def _contract_state(self, original_question: str, current_state: str, node_results: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """收缩状态为新的马尔可夫原子状态"""
        self._aot_logger.info(f"收缩状态: {current_state}")
        
        messages = self._contract_protocol.process_input(
            original_question=original_question,
            current_state=current_state,
            node_results=node_results
        )
        
        response = await self._client.agenerate_content_with_messages(messages, **self.llm_config)
        new_state = self._contract_protocol.parse_output(response)
        
        if not new_state:
            self._aot_logger.warning("状态收缩失败")
            return False, current_state
            
        self._aot_logger.info(f"状态收缩成功，新状态: {new_state}")
        return True, new_state

    def _update_memory_buffer(self, important_facts: List[str]) -> None:
        """更新记忆缓冲区"""
        if not self._enable_memory:
            return
            
        # 为每个事实计算权重
        weighted_facts = []
        for i, fact in enumerate(self._memory_buffer):
            weight = self._memory_weight_decay ** (len(self._memory_buffer) - i - 1)
            weighted_facts.append((fact, weight))
            
        # 添加新事实
        for fact in important_facts:
            if fact not in [f[0] for f in weighted_facts]:
                weighted_facts.append((fact, 1.0))
                
        # 按权重排序并保留最重要的记忆
        weighted_facts.sort(key=lambda x: x[1], reverse=True)
        self._memory_buffer = [fact for fact, _ in weighted_facts[:self._memory_buffer_size]]
            
        self._aot_logger.info(f"更新记忆缓冲区，当前记忆: {self._memory_buffer}")

    def _get_augmented_state(self, current_state: str) -> str:
        """获取增强的状态表示"""
        if not self._enable_memory or not self._memory_buffer:
            return current_state
            
        memory_context = "已确认的重要信息:\n" + "\n".join(self._memory_buffer)
        return f"{current_state}\n\n{memory_context}"

    def _extract_important_facts(self, node_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """从节点结果中提取重要事实"""
        important_facts = []
        
        for node_id, result in node_results.items():
            if "answer" in result and result["answer"]:
                fact = f"{result.get('question', '节点'+node_id)}: {result['answer']}"
                important_facts.append(fact)
                
        return important_facts

    @retry_with_backoff()
    async def _ensemble_answers(self, question: str, methods_results: Dict[str, Dict[str, Any]], 
                              chosen_atom_infos: List[AtomRetrievalInfo]) -> Dict[str, Any]:
        """集成多种方法的答案"""
        if not self._enable_ensemble:
            # 如果未启用集成，返回第一个可用的结果
            for method in self._ensemble_methods:
                if method in methods_results and "answer" in methods_results[method]:
                    return methods_results[method]
            return {}
            
        self._aot_logger.info("开始集成多种方法的答案")
        
        messages = self._ensemble_protocol.process_input(
            question=question,
            methods_results=methods_results,
            chosen_atom_infos=chosen_atom_infos
        )
        
        response = await self._client.agenerate_content_with_messages(messages, **self.llm_config)
        success, result = self._ensemble_protocol.parse_output(response)
        
        if not success:
            self._aot_logger.warning("答案集成失败")
            return {}
            
        self._aot_logger.info(f"答案集成成功，选择了 {result.get('best_method', '')} 方法")
        return result

    async def _process_dag_nodes_parallel(self, nodes: List[Dict], current_state: str, 
                                        chosen_atom_infos: List[AtomRetrievalInfo]) -> Dict[str, Dict[str, Any]]:
        """并行处理DAG节点"""
        if not self._enable_parallel or len(nodes) <= 1:
            # 如果未启用并行或只有一个节点，使用顺序处理
            results = {}
            for node in nodes:
                success, result = await self._answer_node_question(node["question"], chosen_atom_infos)
                if success:
                    results[node["id"]] = result
            return results
            
        self._aot_logger.info(f"开始并行处理 {len(nodes)} 个DAG节点")
        
        # 使用AoTPikeRetriever的并行检索功能
        if isinstance(self._retriever, AoTPikeRetriever):
            retrieval_results = await self._retriever.aretrieve_for_dag_nodes_parallel(nodes, current_state, chosen_atom_infos)
        else:
            # 如果不是AoTPikeRetriever，为每个节点单独检索
            retrieval_results = {}
            for node in nodes:
                atom_infos = await self._retriever.aretrieve_atom_info_through_atom(
                    queries=node["question"],
                    retrieve_id=f"node_{node['id']}"
                )
                retrieval_results[node["id"]] = atom_infos
        
        # 并行处理节点答案
        async def process_node(node):
            node_id = node["id"]
            try:
                atom_infos = retrieval_results.get(node_id, [])
                if atom_infos:
                    success, result = await self._answer_node_question(node["question"], atom_infos)
                    if success:
                        return node_id, result
            except Exception as e:
                self._aot_logger.error(f"节点 {node_id} 处理失败: {e}")
            return node_id, None
        
        # 创建任务
        tasks = [process_node(node) for node in nodes]
        
        # 以固定批次大小并行执行任务
        results = {}
        batch_size = min(self._max_parallel_nodes, len(tasks))
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            for node_id, result in batch_results:
                if result is not None:
                    results[node_id] = result
        
        return results

    async def answer(self, qa: BaseQaData, question_idx: int) -> Dict:
        """使用AoT-PIKE混合方式回答问题"""
        thinking = ""
        new_state = ""
        question = qa.get_question(question_idx)
        current_state = question  # 初始状态就是原始问题
        chosen_atom_infos: List[AtomRetrievalInfo] = []
        node_results = {}
        sorted_nodes = []
        # 记录推理过程
        reasoning_process = {
            "original_question": question,
            "iterations": []
        }
        
        # 如果启用插件模式，使用插件流程
        if self._enable_plugin_mode:
            return await self._plugin_mode_answer(question, chosen_atom_infos)
        
        # 主循环
        for iteration in range(self._max_iterations):
            self._aot_logger.info(f"开始第 {iteration+1} 次迭代")
            iteration_info = {
                "iteration": iteration + 1,
                "current_state": current_state,
                "dag": {},
                "node_results": {},
                "new_state": ""
            }
            
            # 1. 检查当前状态是否可以直接回答
            is_answerable, reason = await self._check_if_answerable(current_state, chosen_atom_infos)
            if is_answerable:
                self._aot_logger.info(f"当前状态可以直接回答: {reason}")
                iteration_info["is_answerable"] = True
                iteration_info["reason"] = reason
                reasoning_process["iterations"].append(iteration_info)
                break
                
            # 2. 将当前状态分解为DAG
            success, thinking, dag = await self._decompose_to_dag(current_state, chosen_atom_infos)
            if not success:
                self._aot_logger.warning("DAG分解失败，尝试直接回答")
                iteration_info["dag_decompose_failed"] = True
                reasoning_process["iterations"].append(iteration_info)
                break
                
            iteration_info["dag"] = dag
            
            # 3. 处理DAG中的节点
            nodes = dag.get("nodes", [])
            
            # 按照依赖关系排序节点
            sorted_nodes = self._topological_sort(nodes)
            
            # 并行处理独立节点
            independent_nodes = [node for node in sorted_nodes if not node["dependencies"]]
            if independent_nodes:
                node_results = await self._process_dag_nodes_parallel(independent_nodes, current_state, chosen_atom_infos)
            else:
                node_results = {}
            
            # 处理依赖节点
            dependent_nodes = [node for node in sorted_nodes if node["dependencies"]]
            for node in dependent_nodes:
                # 检查依赖是否已满足
                if all(dep in node_results for dep in node["dependencies"]):
                    success, result = await self._answer_node_question(node["question"], chosen_atom_infos)
                    if success:
                        node_results[node["id"]] = result
            
            iteration_info["node_results"] = node_results
            
            # 如果没有解决任何节点问题，尝试直接回答
            if not node_results:
                self._aot_logger.warning("未能解决任何节点问题，尝试直接回答")
                iteration_info["no_node_solved"] = True
                reasoning_process["iterations"].append(iteration_info)
                break
            
            # 4. 提取重要事实并更新记忆
            important_facts = self._extract_important_facts(node_results)
            self._update_memory_buffer(important_facts)
            
            # 5. 收缩为新的原子状态
            success, new_state = await self._contract_state(question, current_state, node_results)
            if not success:
                self._aot_logger.warning("状态收缩失败，尝试直接回答")
                iteration_info["contract_failed"] = True
                reasoning_process["iterations"].append(iteration_info)
                break
                
            iteration_info["new_state"] = new_state
            reasoning_process["iterations"].append(iteration_info)
            
            # 如果状态没有变化，直接尝试回答
            if new_state == current_state:
                self._aot_logger.info("状态没有变化，尝试直接回答")
                break
            
            current_state = new_state
        
        # 生成最终答案
        methods_results = {}
        
        # 直接回答
        if "direct" in self._ensemble_methods:
            success, result = await self._answer_node_question(current_state, chosen_atom_infos)
            if success:
                methods_results["direct"] = result
        
        # 分解回答
        if "decompose" in self._ensemble_methods and "node_results" in locals():
            methods_results["decompose"] = {
                "answer": node_results[sorted_nodes[-1]["id"]]["answer"] if node_results and sorted_nodes else "",
                "thinking": thinking
            }
        
        # 收缩回答
        if "contract" in self._ensemble_methods and "new_state" in locals():
            success, result = await self._answer_node_question(new_state, chosen_atom_infos)
            if success:
                methods_results["contract"] = result
        
        # 集成答案
        final_answer = await self._ensemble_answers(question, methods_results, chosen_atom_infos)
        
        # 添加推理过程到结果
        final_answer["reasoning_process"] = reasoning_process
        final_answer["final_state"] = current_state
        
        return final_answer

    async def _plugin_mode_answer(self, question: str, chosen_atom_infos: List[AtomRetrievalInfo]) -> Dict[str, Any]:
        """使用插件模式回答问题"""
        self._aot_logger.info("使用插件模式回答问题")
        
        # 创建多个样本
        sample_results = []
        for i in range(self._plugin_samples):
            self._aot_logger.info(f"处理样本 {i+1}/{self._plugin_samples}")
            
            # 分解问题
            success, thinking, dag = await self._decompose_to_dag(question, chosen_atom_infos)
            if not success:
                continue
                
            # 处理节点
            nodes = dag.get("nodes", [])
            sorted_nodes = self._topological_sort(nodes)
            node_results = await self._process_dag_nodes_parallel(sorted_nodes, question, chosen_atom_infos)
            
            if not node_results:
                continue
                
            # 收缩状态
            success, new_state = await self._contract_state(question, question, node_results)
            if not success:
                continue
                
            # 回答收缩后的问题
            success, result = await self._answer_node_question(new_state, chosen_atom_infos)
            if success:
                result["contracted_question"] = new_state
                result["decomposition"] = dag
                result["node_results"] = node_results
                sample_results.append(result)
        
        if not sample_results:
            self._aot_logger.warning("所有样本处理失败")
            return {}
        
        # 选择最佳结果
        best_result = max(sample_results, key=lambda x: len(x.get("evidence", [])))
        
        return {
            "answer": best_result["answer"],
            "contracted_question": best_result["contracted_question"],
            "thinking": best_result.get("thinking", ""),
            "evidence": best_result.get("evidence", []),
            "samples_count": len(sample_results)
        }

    def _topological_sort(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对DAG节点进行拓扑排序"""
        # 构建邻接表
        graph = {}
        for node in nodes:
            node_id = node["id"]
            graph[node_id] = node["dependencies"]
        
        # 计算入度
        in_degree = {node_id: 0 for node_id in graph}
        for node_id in graph:
            for dep in graph[node_id]:
                in_degree[dep] = in_degree.get(dep, 0) + 1
        
        # 拓扑排序
        queue = [node_id for node_id in graph if in_degree[node_id] == 0]
        sorted_ids = []
        
        while queue:
            node_id = queue.pop(0)
            sorted_ids.append(node_id)
            
            for dep in graph[node_id]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        
        # 检查是否有环
        if len(sorted_ids) != len(graph):
            self._aot_logger.warning("DAG中存在环")
            return nodes  # 如果有环，返回原始顺序
        
        # 根据排序后的ID重新排序节点
        node_map = {node["id"]: node for node in nodes}
        return [node_map[node_id] for node_id in sorted_ids]

    async def run_test_async(self) -> Dict[str, float]:
        """异步运行测试并返回评估结果"""
        # 创建输出目录
        os.makedirs(self._yaml_config["log_dir"], exist_ok=True)
        
        # 创建输出文件
        test_jsonl_path = os.path.join(
            self._yaml_config["log_dir"], 
            f"{self._yaml_config.get('test_jsonl_filename', self._yaml_config['experiment_name'])}.jsonl"
        )
        fout = jsonlines.open(test_jsonl_path, "w")
        
        # 设置默认元数据
        default_metadata = {
            "task": "hotpotqa",
            "model": "aot_pike",
            "description": "AoT-PIKE测试"
        }
        
        # 如果配置中没有元数据，使用默认值
        if "metadata" not in self._yaml_config:
            self._yaml_config["metadata"] = default_metadata
        
        metadata = self._yaml_config["metadata"]
        
        self._aot_logger.info(f"开始测试: {metadata.get('description', '')}")
        self._aot_logger.info(f"测试数据集大小: {len(self._testing_suite)}")
        
        # 运行测试
        for round_idx in range(self._yaml_config.get("test_rounds", 1)):
            round_id = f"Round{round_idx}"
            self._evaluator.on_round_test_start(round_id)
            
            question_idx = 0
            pbar = tqdm(self._testing_suite, desc=f"[{self._yaml_config['experiment_name']}] Round {round_idx}")
            
            for qa in pbar:
                try:
                    # 异步回答问题
                    output_dict = await self.answer(qa, question_idx)
                    
                    # 确保输出包含答案
                    assert "answer" in output_dict, "`answer` should be included in output_dict"
                    answer = output_dict.pop("answer")
                    qa.update_answer(answer)
                    qa.answer_metadata.update(output_dict)
                    
                    # 更新评估指标
                    self._evaluator.update_round_metrics(qa)
                    
                    # 写入结果
                    fout.write(qa.as_dict())
                    
                    # 更新进度条
                    question_idx += 1
                    self._update_pbar_desc(pbar, round_idx=round_idx, count=question_idx)
                    
                except Exception as e:
                    self._aot_logger.error(f"处理问题 {question_idx} 时出错: {str(e)}")
            
            # 完成当前轮次
            self._evaluator.on_round_test_end(round_id)
        
        # 完成测试
        self._evaluator.on_test_end()
        fout.close()
        
        # 收集评估结果
        results = {}
        for metric in self._evaluator._metrics:
            if len(metric._round_scores) > 0:
                results[metric.name] = float(np.mean(metric._round_scores))
        
        return results
    
    def run_test(self) -> Dict[str, float]:
        """运行测试并返回评估结果"""
        # 创建事件循环并运行异步测试
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_test_async()) 