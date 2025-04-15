import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"

from pikerag.workflows.aot_pike_workflow import AoTPikeWorkflow
from pikerag.utils.config_loader import load_dot_env, load_class, load_callable, load_protocol

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 添加默认元数据
    if "metadata" not in config:
        config_name = Path(config_path).stem
        config["metadata"] = {
            "task": "hotpotqa",
            "model": config_name,
            "description": f"{config_name}测试"
        }
    
    return config

def run_test_with_config(config_path):
    """运行单个配置的测试"""
    print(f"\n正在测试配置: {config_path}")
    print("-" * 50)
    
    try:
        config = load_config(config_path)
        workflow = AoTPikeWorkflow(config)
        results = workflow.run_test()
        return results
    except Exception as e:
        print(f"测试失败，错误信息: {str(e)}")
        traceback.print_exc()
        return None

def main():
    # 加载环境变量
    load_dotenv()
    
    # 设置配置文件目录
    config_dir = Path("examples/hotpotqa/configs")
    
    # 获取所有AoT-PIKE相关的配置文件
    aot_configs = list(config_dir.glob("aot_pike*.yml"))
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 准备存储结果
    all_results = []
    
    # 运行每个配置的测试
    for config_path in aot_configs:
        config_name = config_path.stem
        try:
            results = run_test_with_config(config_path)
            if results:
                results['config_name'] = config_name
                all_results.append(results)
                print(f"\n{config_name} 测试完成!")
                print("结果:", results)
        except Exception as e:
            print(f"\n{config_name} 测试失败!")
            print(f"错误信息: {str(e)}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为CSV
    results_df = pd.DataFrame(all_results)
    csv_path = results_dir / f"test_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    
    # 保存为JSON
    json_path = results_dir / f"test_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总结果
    print("\n\n测试汇总")
    print("=" * 50)
    print(f"总共测试配置数: {len(aot_configs)}")
    print(f"成功完成测试数: {len(all_results)}")
    print(f"\n详细结果已保存至:")
    print(f"CSV文件: {csv_path}")
    print(f"JSON文件: {json_path}")
    
    # 打印对比表格
    print("\n性能对比:")
    print(results_df.to_string())

if __name__ == "__main__":
    main() 