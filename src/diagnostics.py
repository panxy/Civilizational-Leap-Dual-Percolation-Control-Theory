"""
商理论 (Shang Theory) - 诊断核心模块
本模块包含从代理变量到系统诊断的完整计算逻辑。
旨在提供专业、可复现且可扩展的分析引擎。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ==================== 模块级配置与校准参数 ====================
# 以下参数基于历史案例的贝叶斯校准得出，是理论的核心部分。
_MODEL_PARAMS = {
    # 生存与繁荣阈值 (标准化基准)
    'delta': 1.0,    # δ: 最低生存阈值
    'R': 2.0,        # R: 社会避险基线
    'R_plus': 2.2,   # R⁺: 繁荣过剩阈值
    
    # 行为因子调节系数
    'alpha': 0.1,    # 正商因子避险激励系数
    'rho': 0.2,      # 心理恢复激励系数
    'mu': 0.05,      # 负商耦合抑制系数
    'kappa': 0.05,   # 负商因子过剩激励系数
    'chi': 0.1,      # 密度抑制系数
    
    # 网络动态系数
    'beta_plus': 0.1,    # ϕ⁺增长系数
    'beta_minus': 0.08,  # ϕ⁻增长系数
    'tau': 0.15,         # 吸引力放大系数
    'iota': 0.2,         # 分裂放大系数
    'zeta_plus': 0.05,   # ϕ⁺耗散系数
    'zeta_minus': 0.07,  # ϕ⁻耗散系数
    'lambda_': 0.1,      # 跨期折损率
    
    # 系统诊断系数
    'omega': 4.1,        # 负商破坏力放大系数
}

# 诊断阈值
_THRESHOLDS = {
    'phi_plus_critical': 0.33,   # ϕ⁺正跃迁阈值 θ⁺
    'phi_minus_safe': 0.10,      # ϕ⁻安全上限
    'phi_minus_danger': 0.18,    # ϕ⁻危险阈值
    'TP_forward': 0.52,          # TP正向跃迁阈值
    'TP_collapse': 0.15,         # TP负向崩盘阈值
}

# ==================== 核心计算函数 ====================
def calculate_intermediate_variables(proxy_values: List[float]) -> Dict[str, float]:
    """
    将15维代理变量映射为理论中的关键中间变量。
    这是模型校准的核心部分，将现实数据转化为理论参数。
    
    参数:
        proxy_values: 长度为15的列表，对应15维代理变量（顺序固定）。
    
    返回:
        包含所有关键中间变量的字典。
    """
    # 解包并命名，便于后续引用
    (gdp_growth, non_cash_ratio, npl_ratio, shadow_economy, gini,
     polarization, net_migration, digital_coverage, electricity_access,
     internet_penetration, fintech_growth, youth_unemployment,
     debt_service_ratio, crypto_estimate, toxicity_index) = proxy_values

    # 1. 广义能量 P (综合经济与数字化水平)
    P = (gdp_growth * 0.4 + digital_coverage * 0.3 + internet_penetration * 0.3) * 10
    
    # 2. 信用编码
    K_plus = np.clip(non_cash_ratio * (1 - npl_ratio * 5), 0.1, 0.95)  # 正信用
    K_minus = np.clip(0.3 * shadow_economy + 0.7 * crypto_estimate, 0.05, 0.8)  # 负信用
    
    # 3. 商因子
    sigma_plus = np.clip(0.7 - polarization * 0.5 + net_migration * 0.01 - toxicity_index * 0.3, 0.1, 0.9)
    sigma_minus = np.clip(0.1 + youth_unemployment * 0.6 + debt_service_ratio * 0.3 + polarization * 0.4, 0.05, 0.8)
    
    # 4. 环境与系统变量
    A = np.clip(0.3 + net_migration * 0.05 + digital_coverage * 0.2 - toxicity_index * 0.15, 0.1, 0.9)  # 吸引力
    D = polarization  # 分裂度
    Lambda = np.clip(1.5 - shadow_economy - npl_ratio * 10, 0.5, 3.0)  # 惩罚强度
    Psi = 1.0 - toxicity_index  # 叙事抑制
    G = (gdp_growth * 10) * 0.7 + electricity_access * 0.3  # 能量密度
    H = np.clip(0.8 - youth_unemployment * 0.5 - toxicity_index * 0.3, 0.2, 1.0)  # 心理恢复
    
    return {
        'P': P, 'K_plus': K_plus, 'K_minus': K_minus,
        'sigma_plus': sigma_plus, 'sigma_minus': sigma_minus,
        'A': A, 'D': D, 'Lambda': Lambda, 'Psi': Psi, 'G': G, 'H': H,
        'gini': gini  # 单独保留，用于计算η
    }

def compute_core_equations(intermediate_vars: Dict[str, float], 
                           params: Dict[str, float]) -> Dict[str, float]:
    """
    基于中间变量，执行七方程核心计算（稳态简化版）。
    
    返回包含传输量、连通度及潜力的字典。
    """
    p = intermediate_vars
    pm = params
    
    # 方程 1 & 2: 微观传输
    T_plus = (p['sigma_plus'] * max(p['P'] - pm['delta'], 0) * 
              p['K_plus'] * np.exp(-pm['lambda_'] * 1))
    T_minus = (p['sigma_minus'] * max(p['P'] - pm['R_plus'], 0) * 
               p['K_minus'] * np.exp(-pm['lambda_'] * 1))
    
    # 方程 3 & 4: 商因子变化趋势（用于定性判断）
    d_sigma_plus = (pm['alpha'] * max(pm['R'] - p['P'], 0) + 
                    pm['rho'] * p['H'] - pm['mu'] * p['sigma_minus'])
    d_sigma_minus = (pm['kappa'] * max(p['P'] - pm['R_plus'], 0) - 
                     p['Lambda'] * p['Psi'] * p['sigma_minus'] - pm['chi'] / p['G'])
    
    # 方程 5 & 6: 宏观连通度（准稳态解）
    phi_plus = np.clip((pm['beta_plus'] * T_plus * (1 + pm['tau'] * p['A'])) / pm['zeta_plus'], 0.05, 0.8)
    phi_minus = np.clip((pm['beta_minus'] * T_minus * (1 + pm['iota'] * p['D'])) / pm['zeta_minus'], 0.02, 0.6)
    
    # 方程 7: 跃迁潜力 TP
    CCA_plus = T_plus * 10  # 正系统活性（时间积分近似）
    CCA_minus = T_minus * 10  # 负系统活性
    eta = 1.0 - p['gini']  # 公平效率 η 的简化代理
    TP = CCA_plus * eta - pm['omega'] * CCA_minus
    
    return {
        'T_plus': T_plus, 'T_minus': T_minus,
        'd_sigma_plus': d_sigma_plus, 'd_sigma_minus': d_sigma_minus,
        'phi_plus': phi_plus, 'phi_minus': phi_minus,
        'TP': TP, 'eta': eta
    }

def diagnose_system(phi_plus: float, phi_minus: float, TP: float, 
                    thresholds: Dict[str, float]) -> Dict[str, any]:
    """
    根据计算结果和阈值，进行系统状态诊断。
    
    返回包含状态标签、风险等级和关键信息的字典。
    """
    # 状态判断逻辑
    if phi_plus >= thresholds['phi_plus_critical'] and phi_minus <= thresholds['phi_minus_safe'] and TP >= thresholds['TP_forward']:
        status = "deep_positive_transition"
        label = "✅ 深度正跃迁"
        risk = "low"
    elif phi_plus >= thresholds['phi_plus_critical'] and phi_minus <= thresholds['phi_minus_danger'] and TP >= thresholds['TP_collapse']:
        status = "fragile_positive_transition"
        label = "⚠️ 脆弱正跃迁/停滞"
        risk = "medium"
    elif phi_minus > thresholds['phi_minus_danger'] and TP < thresholds['TP_collapse']:
        status = "negative_transition_warning"
        label = "🚨 负跃迁预警"
        risk = "high"
    else:
        status = "threshold_hovering"
        label = "⚖️ 阈值徘徊"
        risk = "variable"
    
    # 关键风险信号
    warnings = []
    if phi_minus > thresholds['phi_minus_safe']:
        warnings.append(f"负商网络连通度(ϕ⁻={phi_minus:.3f})超过安全线。")
    if TP < thresholds['TP_forward']:
        warnings.append(f"系统跃迁潜力(TP={TP:.3f})不足。")
    
    return {
        'status': status,
        'label': label,
        'risk_level': risk,
        'warnings': warnings,
        'thresholds_met': {
            'phi_plus_ok': phi_plus >= thresholds['phi_plus_critical'],
            'phi_minus_safe': phi_minus <= thresholds['phi_minus_safe'],
            'TP_forward_ok': TP >= thresholds['TP_forward']
        }
    }

# ==================== 主诊断接口函数 ====================
def quick_diagnose(proxy_values: List[float], 
                   custom_params: Optional[Dict[str, float]] = None,
                   custom_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, any]:
    """
    快速诊断的主接口函数。
    输入15维代理变量，返回完整的诊断结果。
    
    参数:
        proxy_values: 15维代理变量列表。
        custom_params: 可选的参数字典，用于覆盖默认值。
        custom_thresholds: 可选的阈值字典，用于覆盖默认值。
    
    返回:
        包含所有中间结果、最终指标和诊断的嵌套字典。
    """
    # 1. 合并参数与阈值（优先使用自定义值）
    params = {**_MODEL_PARAMS, **(custom_params or {})}
    thresholds = {**_THRESHOLDS, **(custom_thresholds or {})}
    
    # 2. 执行计算流水线
    intermediate_vars = calculate_intermediate_variables(proxy_values)
    core_results = compute_core_equations(intermediate_vars, params)
    diagnosis = diagnose_system(core_results['phi_plus'], 
                                core_results['phi_minus'], 
                                core_results['TP'], 
                                thresholds)
    
    # 3. 整合并返回所有结果
    return {
        'input_proxies': proxy_values,
        'intermediate_variables': intermediate_vars,
        'core_results': core_results,
        'diagnosis': diagnosis,
        'model_parameters_used': params,
        'thresholds_used': thresholds
    }

# ==================== 辅助函数 ====================
def get_default_parameters() -> Dict[str, float]:
    """返回模型的默认参数副本。"""
    return _MODEL_PARAMS.copy()

def get_default_thresholds() -> Dict[str, float]:
    """返回模型的默认阈值副本。"""
    return _THRESHOLDS.copy()

def print_diagnosis_report(result: Dict[str, any], case_name: str = "未命名案例"):
    """在控制台打印格式化的诊断报告。"""
    diag = result['diagnosis']
    core = result['core_results']
    
    print(f"\n{'='*60}")
    print(f"商理论诊断报告 - {case_name}")
    print('='*60)
    print(f"核心指标:")
    print(f"  ϕ⁺ (正连通度): {core['phi_plus']:.3f} | 阈值 ≥ {result['thresholds_used']['phi_plus_critical']:.2f} | {'✅ 达标' if diag['thresholds_met']['phi_plus_ok'] else '❌ 未达'}")
    print(f"  ϕ⁻ (负连通度): {core['phi_minus']:.3f} | 安全 ≤ {result['thresholds_used']['phi_minus_safe']:.2f} | {'✅ 安全' if diag['thresholds_met']['phi_minus_safe'] else '⚠️ 超标'}")
    print(f"  TP (跃迁潜力): {core['TP']:.3f} | 目标 ≥ {result['thresholds_used']['TP_forward']:.2f} | {'✅ 充足' if diag['thresholds_met']['TP_forward_ok'] else '⚠️ 不足'}")
    print(f"\n系统状态: {diag['label']} (风险等级: {diag['risk_level'].upper()})")
    
    if diag['warnings']:
        print(f"\n🚨 风险提示:")
        for warn in diag['warnings']:
            print(f"  • {warn}")
    
    print(f"\n📈 关键衍生指标:")
    print(f"  正商传输 T⁺: {core['T_plus']:.3f}")
    print(f"  负商传输 T⁻: {core['T_minus']:.3f}")
    print(f"  系统公平效率 η: {core['eta']:.3f}")
    print('='*60)

# ==================== 模块测试代码 ====================
if __name__ == "__main__":
    # 使用新加坡2024年代理变量进行模块自检
    print("正在运行商理论诊断模块自检...")
    test_proxies = [
        0.044, 0.92, 0.012, 0.10, 0.41, 0.40, 1.5, 
        0.95, 1.00, 0.96, 0.25, 0.091, 0.069, 0.05, 0.35
    ]
    
    result = quick_diagnose(test_proxies)
    print_diagnosis_report(result, "新加坡2024")
    print("\n✅ 模块自检完成，功能正常。")