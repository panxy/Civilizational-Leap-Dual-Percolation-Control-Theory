"""
商理论 (Shang Theory) - 实用工具函数
包含数据加载、结果可视化等辅助功能。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_proxy_data(filepath):
    """
    加载15维代理变量的CSV数据文件。
    
    参数:
    filepath (str): CSV文件路径，例如 'data/global_2024_backtest.csv'
    
    返回:
    pandas.DataFrame: 包含代理变量数据的数据框
    """
    try:
        df = pd.read_csv(filepath)
        print(f"数据加载成功，共 {df.shape[0]} 行， {df.shape[1]} 列。")
        return df
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None

def plot_transition_scatter(phi_plus, phi_minus, labels=None, save_path=None):
    """
    绘制ϕ⁺与ϕ⁻的状态空间散点图，并标注理论阈值区域。
    
    参数:
    phi_plus (list or np.ndarray): 正连通度数组
    phi_minus (list or np.ndarray): 负连通度数组
    labels (list, optional): 每个数据点的标签（如案例名称）
    save_path (str, optional): 图片保存路径，如 'docs/transition_plot.png'
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制散点
    scatter = plt.scatter(phi_plus, phi_minus, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # 标注阈值线
    plt.axvline(x=0.33, color='green', linestyle='--', alpha=0.7, label='ϕ⁺ 正跃迁阈值 (0.33)')
    plt.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='ϕ⁻ 安全上限 (0.10)')
    plt.axhline(y=0.18, color='red', linestyle=':', alpha=0.7, label='ϕ⁻ 危险阈值 (0.18)')
    
    # 填充正跃迁安全区 (ϕ⁺ > 0.33 & ϕ⁻ < 0.10)
    plt.fill_betweenx([0, 0.10], 0.33, 1, color='green', alpha=0.1, label='正跃迁区')
    # 填充高风险区 (ϕ⁻ > 0.18)
    plt.fill_between([0, 1], 0.18, 1, color='red', alpha=0.05, label='高风险区')
    
    # 添加标签（如果有）
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (phi_plus[i], phi_minus[i]), fontsize=8, alpha=0.8)
    
    plt.xlabel('正商网络连通度 (ϕ⁺)', fontsize=12)
    plt.ylabel('负商网络连通度 (ϕ⁻)', fontsize=12)
    plt.title('商理论：文明系统状态空间图', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    
    plt.show()

def simulate_time_series(steps, initial_conditions, params):
    """
    一个简单的时间序列模拟示例框架。
    你可以根据7个方程的耦合关系扩展此函数。
    
    参数:
    steps (int): 模拟的时间步数
    initial_conditions (dict): 初始状态变量字典
    params (dict): 模型参数字典
    
    返回:
    pandas.DataFrame: 包含各变量时间序列的数据框
    """
    # 此函数是一个框架，你需要根据微分方程组的实际关系进行实现
    print("时间序列模拟框架 - 需要根据实际方程耦合关系进行实现")
    # 提示：可以考虑使用欧拉法或SciPy的ODE求解器进行实现
    
    # 示例返回一个空DataFrame
    return pd.DataFrame()

if __name__ == "__main__":
    print("工具函数模块导入成功。")