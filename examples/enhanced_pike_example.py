"""
增强版PIKE示例 - 展示如何使用集成了miniRAG异构图功能的PIKE检索器和工作流
"""

import os
import asyncio
from pathlib import Path
import yaml

from pikerag.workflows.enhanced_aot_pike_workflow import EnhancedAoTPikeWorkflow
from pikerag.knowledge_retrievers.enhanced_pike_retriever import EnhancedPikeRetriever
from pikerag.utils.config_loader import load_workflow_from_yaml
from pikerag.utils.logger import Logger


class SimpleQaData:
    """简单的QA数据类，用于测试"""
    
    def __init__(self, questions, documents=None):
        self.questions = questions
        self.documents = documents
        
    def get_question(self, idx):
        return self.questions[idx]


async def main():
    # 设置日志
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    logger = Logger("enhanced_pike_example", log_dir=str(log_dir))
    
    # 加载配置
    config_path = Path("pikerag/workflows/templates/EnhancedAoTPikeWorkflow.yml")
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
    
    # 创建工作流
    workflow = load_workflow_from_yaml(yaml_config, log_dir=str(log_dir), main_logger=logger)
    
    # 确保工作流是EnhancedAoTPikeWorkflow
    if not isinstance(workflow, EnhancedAoTPikeWorkflow):
        logger.error(f"工作流不是EnhancedAoTPikeWorkflow: {type(workflow)}")
        return
        
    # 确保检索器是EnhancedPikeRetriever
    if not isinstance(workflow._retriever, EnhancedPikeRetriever):
        logger.error(f"检索器不是EnhancedPikeRetriever: {type(workflow._retriever)}")
        return
    
    # 示例文档
    documents = [
        """
        人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
        
        人工智能的主要研究领域包括：
        1. 机器学习：使计算机能够自动从数据中学习，不需要明确编程。
        2. 自然语言处理：使计算机能够理解和生成人类语言。
        3. 计算机视觉：使计算机能够从图像或视频中获取信息。
        4. 专家系统：模拟人类专家的决策能力。
        5. 机器人学：研究如何设计、制造和应用机器人。
        
        深度学习是机器学习的一个分支，它基于人工神经网络，通过多层次的抽象来学习数据的表示。
        深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。
        
        大型语言模型（Large Language Models，简称LLM）是一种基于深度学习的自然语言处理模型，
        它通过在大量文本数据上训练，能够生成连贯、流畅且符合上下文的文本。
        著名的大型语言模型包括GPT（Generative Pre-trained Transformer）系列、BERT（Bidirectional Encoder Representations from Transformers）等。
        
        人工智能的发展面临许多挑战，包括伦理问题、隐私问题、安全问题等。
        如何确保人工智能的发展符合人类的价值观和利益，是当前人工智能研究中的重要议题。
        """,
        
        """
        知识图谱（Knowledge Graph）是一种结构化的知识表示方式，它以图的形式存储实体和它们之间的关系。
        知识图谱由节点（表示实体）和边（表示关系）组成，可以表示复杂的知识网络。
        
        知识图谱的主要组成部分包括：
        1. 实体（Entity）：知识图谱中的节点，表示现实世界中的对象或概念。
        2. 关系（Relation）：知识图谱中的边，表示实体之间的联系。
        3. 属性（Attribute）：实体的特征或性质。
        
        知识图谱的构建通常包括以下步骤：
        1. 实体抽取：从文本中识别和提取实体。
        2. 关系抽取：识别实体之间的关系。
        3. 实体链接：将抽取的实体链接到已有的知识库中。
        4. 知识融合：整合来自不同来源的知识。
        
        知识图谱在搜索引擎、推荐系统、问答系统等领域有广泛应用。
        例如，Google的知识图谱可以在搜索结果中直接显示相关实体的信息，
        而不仅仅是提供网页链接。
        
        异构图（Heterogeneous Graph）是一种特殊类型的图，其中节点和边可以有不同的类型。
        在知识图谱中，异构图可以表示不同类型的实体（如人物、组织、地点）和它们之间的不同类型的关系。
        
        图神经网络（Graph Neural Networks，简称GNN）是一种专门用于处理图结构数据的深度学习模型。
        它可以学习节点的表示，并利用图的结构信息进行预测。
        在知识图谱中，图神经网络可以用于链接预测、节点分类等任务。
        """
    ]
    
    # 示例问题
    questions = [
        "什么是人工智能？它的主要研究领域有哪些？",
        "知识图谱是什么？它与异构图有什么关系？"
    ]
    
    # 创建QA数据
    qa_data = SimpleQaData(questions, documents)
    
    # 处理文档
    logger.info("开始处理文档...")
    for i, doc in enumerate(documents):
        success = await workflow.process_document(doc, f"doc_{i}")
        logger.info(f"文档 {i} 处理{'成功' if success else '失败'}")
    
    # 回答问题
    for i, question in enumerate(questions):
        logger.info(f"问题 {i+1}: {question}")
        answer = await workflow.answer(qa_data, i)
        logger.info(f"回答: {answer.get('answer', '无法回答')}")
        logger.info("-" * 50)
    
    # 获取统计信息
    stats = workflow._retriever.get_statistics()
    logger.info(f"检索器统计信息: {stats}")


if __name__ == "__main__":
    asyncio.run(main()) 