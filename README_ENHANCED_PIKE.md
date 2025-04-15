# 增强版PIKE-RAG：集成miniRAG的异构图功能

本项目将miniRAG的异构图构建和查询功能集成到PIKE-RAG中，保持了PIKE-RAG的动态更新能力，同时增强了知识表示和检索能力。

## 功能特点

1. **异构图构建**：使用miniRAG的实体抽取和关系构建功能，构建显式的知识图谱
2. **融合检索**：结合PIKE-RAG的原子查询检索和miniRAG的图检索功能
3. **动态更新**：保持PIKE-RAG的动态原子查询生成和图更新能力
4. **多种存储后端**：支持多种图存储后端，如NetworkX、Neo4j等

## 安装依赖

首先，确保安装了PIKE-RAG和miniRAG的依赖：

```bash
# 安装PIKE-RAG依赖
pip install -r requirements.txt

# 安装miniRAG依赖
pip install minirag
```

## 目录结构

```
pikerag/
├── knowledge_retrievers/
│   ├── adapters/
│   │   └── minirag_adapter.py  # miniRAG适配器
│   ├── enhanced_pike_retriever.py  # 增强版PIKE检索器
│   └── templates/
│       └── EnhancedPikeRetriever.yml  # 配置模板
├── workflows/
│   ├── enhanced_aot_pike_workflow.py  # 增强版工作流
│   └── templates/
│       └── EnhancedAoTPikeWorkflow.yml  # 配置模板
└── ...

examples/
└── enhanced_pike_example.py  # 使用示例
```

## 使用方法

### 1. 配置

在配置文件中设置增强版PIKE检索器和工作流：

```yaml
# 工作流配置
workflow:
  module_path: pikerag.workflows
  class_name: EnhancedAoTPikeWorkflow
  args:
    # 工作流参数
    graph_search_depth: 2  # 图搜索深度

# 检索器配置
retriever:
  module_path: pikerag.knowledge_retrievers
  class_name: EnhancedPikeRetriever
  args:
    # miniRAG集成配置
    minirag_config:
      enable_minirag: true
      enable_entity_extraction: true
      enable_graph_query: true
      fusion_mode: "hybrid"  # hybrid, pike_only, graph_only
      
      # 图存储配置
      graph_storage:
        storage_type: "NetworkXStorage"
        module_path: "minirag.kg.networkx_impl"
        namespace: "pikerag"
```

### 2. 处理文档

```python
# 初始化工作流
workflow = load_workflow_from_yaml(yaml_config, log_dir=str(log_dir), main_logger=logger)

# 处理文档
async def process_documents(documents):
    for i, doc in enumerate(documents):
        success = await workflow.process_document(doc, f"doc_{i}")
        print(f"文档 {i} 处理{'成功' if success else '失败'}")
```

### 3. 回答问题

```python
# 创建QA数据
qa_data = SimpleQaData(questions, documents)

# 回答问题
async def answer_questions():
    for i, question in enumerate(questions):
        print(f"问题: {question}")
        answer = await workflow.answer(qa_data, i)
        print(f"回答: {answer.get('answer', '无法回答')}")
```

## 工作原理

### 异构图构建流程

1. **文档处理**：将文档分块，并添加到PIKE-RAG的chunk_store中
2. **实体抽取**：使用miniRAG的实体抽取功能，从文本中抽取实体和关系
3. **图更新**：将抽取的实体和关系添加到图存储中
4. **原子查询生成**：从实体和关系生成原子查询，并更新PIKE-RAG的atom_store

### 融合检索流程

1. **PIKE检索**：使用PIKE-RAG的原子查询检索功能
2. **图检索**：使用miniRAG的图检索功能
3. **结果融合**：合并两种检索结果，并根据相关性排序
4. **动态更新**：如果检索结果不足，动态生成新的原子查询和实体关系

## 示例

参见 `examples/enhanced_pike_example.py` 文件，展示了如何使用增强版PIKE-RAG处理文档和回答问题。

## 注意事项

1. 确保miniRAG模块可用，否则增强功能将不可用
2. 图存储需要足够的内存，特别是对于大型知识图谱
3. 实体抽取和关系构建依赖于LLM的质量，建议使用高质量的LLM

## 自定义

### 自定义图存储

可以通过修改配置文件中的 `graph_storage` 部分来使用不同的图存储后端：

```yaml
graph_storage:
  storage_type: "Neo4JStorage"  # 使用Neo4j作为图存储
  module_path: "minirag.kg.neo4j_impl"
  namespace: "pikerag"
  global_config:
    # Neo4j特定配置
```

### 自定义融合模式

可以通过修改 `fusion_mode` 参数来控制检索模式：

- `hybrid`：同时使用PIKE检索和图检索
- `pike_only`：仅使用PIKE检索
- `graph_only`：仅使用图检索 