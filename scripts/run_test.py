import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pikerag.workflows import AoTPikeWorkflow
from pikerag.utils.config_loader import load_config

def main():
    # 加载环境变量
    load_dotenv()
    
    # 加载配置文件
    config_path = "examples/hotpotqa/configs/aot_pike.yml"
    config = load_config(config_path)
    
    # 创建工作流实例
    workflow = AoTPikeWorkflow(config)
    
    # 运行测试
    results = workflow.run_test()
    
    # 输出结果
    print("\n测试结果:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 