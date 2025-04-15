# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Any
import asyncio

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.utils.json_parser import parse_json


DEFAULT_SYSTEM_PROMPT = "你是一个擅长问题分解和推理的AI助手。"


def atom_infos_to_context_string(chosen_atom_infos: List[AtomRetrievalInfo], limit: int=80000) -> str:
    """将选择的原子信息转换为上下文字符串"""
    context: str = ""
    chunk_id_set = set()
    for info in chosen_atom_infos:
        if info.source_chunk_id in chunk_id_set:
            continue
        chunk_id_set.add(info.source_chunk_id)

        if info.source_chunk_title is not None:
            context += f"\n标题: {info.source_chunk_title}. 内容: {info.source_chunk}\n"
        else:
            context += f"\n{info.source_chunk}\n"

        if len(context) >= limit:
            break

    context = context.strip()
    return context

################################################################################
# DAG分解提示模板

aot_dag_decomposition_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
分析给定问题，将其分解为一个依赖型有向无环图(DAG)。每个子问题应该是原子性的、可独立回答的。

# 输出格式
请以以下JSON格式输出:
{{
    "thinking": "你对问题的分析思考",
    "nodes": [
        {{
            "id": "子问题唯一标识，如q1",
            "question": "子问题内容",
            "dependencies": ["此子问题依赖的其他子问题ID列表"]
        }},
        ...
    ]
}}

# 当前问题
{content}

# 已知上下文
{context_str}

# 你的输出:
""".strip()),
    ],
    input_variables=["content", "context_str"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class AoTDagDecompositionParser(BaseContentParser):
    """DAG分解解析器"""
    
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, dict]:
        """编码输入"""
        context = atom_infos_to_context_string(chosen_atom_infos)
        return content, {"context_str": context}

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        """解码输出"""
        try:
            output = parse_json(content)
            thinking = output.get("thinking", "")
            nodes = output.get("nodes", [])
            
            # 验证节点格式
            for node in nodes:
                if not all(key in node for key in ["id", "question", "dependencies"]):
                    return False, thinking, {}
            
            # 验证依赖关系
            node_ids = {node["id"] for node in nodes}
            for node in nodes:
                for dep in node["dependencies"]:
                    if dep not in node_ids:
                        return False, thinking, {}
            
            return True, thinking, {"nodes": nodes}
        except Exception as e:
            print(f"[AoTDagDecompositionParser] 解析错误: {e}")
            print(f"内容: {content}")
            return False, "", {}


aot_dag_decomposition_protocol = CommunicationProtocol(
    template=aot_dag_decomposition_template,
    parser=AoTDagDecompositionParser(),
)

################################################################################
# 子问题收缩提示模板

aot_contract_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
将已解答的子问题收缩为一个新的马尔可夫原子状态。新状态应该是一个独立的问题，等价于原始问题，但删除了已解决的部分。

# 输出格式
直接输出新的原子问题状态，不包含其他内容。

# 原始问题
{original_question}

# 当前问题状态
{current_state}

# 子问题及其答案
{node_results_str}

# 你的输出:
""".strip()),
    ],
    input_variables=["original_question", "current_state", "node_results_str"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class AoTContractParser(BaseContentParser):
    """子问题收缩解析器"""
    
    def encode(self, original_question: str, current_state: str, node_results: Dict[str, Dict[str, Any]], **kwargs) -> Tuple[str, dict]:
        """编码输入"""
        # 格式化节点结果
        node_results_str = ""
        for node_id, result in node_results.items():
            node_results_str += f"子问题ID: {node_id}\n"
            node_results_str += f"问题: {result.get('question', '')}\n"
            node_results_str += f"答案: {result.get('answer', '')}\n"
            if 'evidence' in result:
                node_results_str += f"证据: {result.get('evidence', '')}\n"
            node_results_str += "\n"
        
        return original_question, {
            "current_state": current_state,
            "node_results_str": node_results_str
        }

    def decode(self, content: str, **kwargs) -> str:
        """解码输出"""
        # 直接返回收缩后的问题
        return content.strip()


aot_contract_protocol = CommunicationProtocol(
    template=aot_contract_template,
    parser=AoTContractParser(),
)

################################################################################
# 节点问题解答提示模板

aot_node_answering_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
根据提供的上下文回答给定的子问题。

# 输出格式
请以以下JSON格式输出:
{{
    "thinking": "你的思考过程",
    "answer": "子问题的答案",
    "evidence": ["支持你答案的关键证据"]
}}

# 子问题
{node_question}

# 上下文
{context_str}

# 你的输出:
""".strip()),
    ],
    input_variables=["node_question", "context_str"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class AoTNodeAnsweringParser(BaseContentParser):
    """节点问题解答解析器"""
    
    def encode(self, node_question: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, dict]:
        """编码输入"""
        context = atom_infos_to_context_string(chosen_atom_infos)
        return node_question, {"context_str": context}

    def decode(self, content: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """解码输出"""
        try:
            output = parse_json(content)
            thinking = output.get("thinking", "")
            answer = output.get("answer", "")
            evidence = output.get("evidence", [])
            
            return True, {
                "thinking": thinking,
                "answer": answer,
                "evidence": evidence
            }
        except Exception as e:
            print(f"[AoTNodeAnsweringParser] 解析错误: {e}")
            print(f"内容: {content}")
            return False, {}


aot_node_answering_protocol = CommunicationProtocol(
    template=aot_node_answering_template,
    parser=AoTNodeAnsweringParser(),
)

################################################################################
# 可解答性检查提示模板

aot_answerability_check_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
评估当前问题状态是否可以直接回答，无需进一步分解。

# 输出格式
请以以下JSON格式输出:
{{
    "thinking": "你的思考过程",
    "is_answerable": true或false,
    "reason": "你的判断理由"
}}

# 当前问题状态
{current_state}

# 已知上下文
{context_str}

# 你的输出:
""".strip()),
    ],
    input_variables=["current_state", "context_str"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class AoTAnswerabilityCheckParser(BaseContentParser):
    """可解答性检查解析器"""
    
    def encode(self, current_state: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, dict]:
        """编码输入"""
        context = atom_infos_to_context_string(chosen_atom_infos)
        return current_state, {"context_str": context}

    def decode(self, content: str, **kwargs) -> Tuple[bool, bool, str]:
        """解码输出"""
        try:
            output = parse_json(content)
            thinking = output.get("thinking", "")
            is_answerable = output.get("is_answerable", False)
            reason = output.get("reason", "")
            
            return True, is_answerable, reason
        except Exception as e:
            print(f"[AoTAnswerabilityCheckParser] 解析错误: {e}")
            print(f"内容: {content}")
            return False, False, ""


aot_answerability_check_protocol = CommunicationProtocol(
    template=aot_answerability_check_template,
    parser=AoTAnswerabilityCheckParser(),
)

################################################################################
# 动态原子查询生成提示模板

dynamic_atom_generation_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
根据给定的子问题和上下文，生成可能有助于回答该子问题的原子查询。原子查询应该是简短、具体的问题，能够直接从知识库中检索相关信息。

# 输出格式
请以以下JSON格式输出:
{{
    "thinking": "你的思考过程",
    "atoms": [
        "原子查询1",
        "原子查询2",
        ...
    ]
}}

# 子问题
{node_question}

# 当前问题状态
{current_state}

# 已知上下文
{context}

# 你的输出:
""".strip()),
    ],
    input_variables=["node_question", "current_state", "context"],
    partial_variables={
        "system_prompt": "你是一个擅长生成精确查询的AI助手。",
    },
)


class DynamicAtomGenerationParser(BaseContentParser):
    """动态原子查询生成解析器"""
    
    def encode(self, node_question: str, current_state: str, context: str, **kwargs) -> Tuple[str, dict]:
        """编码输入"""
        return node_question, {
            "current_state": current_state,
            "context": context
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, List[str]]:
        """解码输出"""
        try:
            output = parse_json(content)
            thinking = output.get("thinking", "")
            atoms = output.get("atoms", [])
            
            # 验证原子查询
            valid_atoms = []
            for atom in atoms:
                if isinstance(atom, str) and atom.strip():
                    valid_atoms.append(atom.strip())
            
            return len(valid_atoms) > 0, valid_atoms
        except Exception as e:
            print(f"[DynamicAtomGenerationParser] 解析错误: {e}")
            print(f"内容: {content}")
            return False, []


dynamic_atom_generation_protocol = CommunicationProtocol(
    template=dynamic_atom_generation_template,
    parser=DynamicAtomGenerationParser(),
)

################################################################################
# 集成决策提示模板

ensemble_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
评估多种方法生成的答案质量，并选择最佳答案或生成一个集成答案。

# 输出格式
请以以下JSON格式输出:
{{
    "thinking": "你的分析思考过程",
    "evaluation": [
        {{
            "method": "方法1",
            "answer": "对应的答案",
            "strengths": ["该答案的优势1", "该答案的优势2", ...],
            "weaknesses": ["该答案的不足1", "该答案的不足2", ...],
            "score": 分数(0-10)
        }},
        ...更多方法评估
    ],
    "best_method": "最佳方法名称",
    "answer": "最终选择的或集成的答案",
    "explanation": "为什么选择这个答案的解释"
}}

# 原始问题
{question}

# 方法结果
{methods_results}

# 上下文信息（如果有）
{context_str}

# 你的输出:
""".strip()),
    ],
    input_variables=["question", "methods_results", "context_str"],
    partial_variables={
        "system_prompt": "你是一个擅长评估答案质量的AI助手。",
    },
)


class EnsembleDecisionParser(BaseContentParser):
    """集成决策解析器"""
    
    def encode(self, question: str, methods_results: Dict[str, Dict[str, Any]], chosen_atom_infos: List[AtomRetrievalInfo] = None, **kwargs) -> Tuple[str, dict]:
        """编码输入"""
        context = ""
        if chosen_atom_infos:
            context = atom_infos_to_context_string(chosen_atom_infos)
            
        # 格式化方法结果
        methods_results_str = ""
        for method_name, result in methods_results.items():
            methods_results_str += f"## {method_name}方法\n"
            methods_results_str += f"答案: {result.get('answer', '无答案')}\n"
            if "thinking" in result:
                methods_results_str += f"思考过程: {result.get('thinking', '')}\n"
            methods_results_str += "\n"
        
        return question, {
            "methods_results": methods_results_str,
            "context_str": context
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """解码输出"""
        try:
            output = parse_json(content)
            thinking = output.get("thinking", "")
            evaluation = output.get("evaluation", [])
            best_method = output.get("best_method", "")
            answer = output.get("answer", "")
            explanation = output.get("explanation", "")
            
            return True, {
                "thinking": thinking,
                "evaluation": evaluation,
                "best_method": best_method,
                "answer": answer,
                "explanation": explanation
            }
        except Exception as e:
            print(f"[EnsembleDecisionParser] 解析错误: {e}")
            print(f"内容: {content}")
            return False, {}


ensemble_protocol = CommunicationProtocol(
    template=ensemble_template,
    parser=EnsembleDecisionParser(),
)

################################################################################
# 错误重试装饰器

def retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """带指数退避的重试装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"[重试] 达到最大重试次数 {max_retries}，操作失败: {e}")
                        raise e
                    
                    print(f"[重试] 第 {retries} 次重试，等待 {delay} 秒: {e}")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            
        return wrapper
    return decorator 