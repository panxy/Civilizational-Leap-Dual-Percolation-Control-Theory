#!/usr/bin/env python3
"""
商理论 (Shang Theory) 快速诊断工具 - 大众版
只需输入15维代理变量，即可得到系统状态诊断。
"""

import numpy as np
import pandas as pd

# ==================== 第一步：用户输入区 ====================
# 请在这里输入或替换你想要诊断的对象的15维代理变量数据
# 以新加坡2024年数据为例（你可以完全替换此字典）
input_data = {
    # 序号: [代理变量名称， 数值, 说明]
    1: ["GDP per capita growth", 0.044, ""],  # 4.4%
    2: ["Non-cash payment transactions / total", 0.92, ""],
    3: ["NPL ratio (bank non-performing loans)", 0.012, ""],  # 1.2%
    4: ["Shadow economy (% of GDP)", 0.10, ""],
    5: ["Gini coefficient", 0.41, ""],
    6: ["Polarization index (0–1)", 0.40, ""],
    7: ["Net migration rate (per 1,000)", 1.5, ""],  # 每千人+1.5
    8: ["Digital infrastructure coverage", 0.95, ""],
    9: ["Electricity access rate", 1.00, ""],
    10: ["Internet penetration", 0.96, ""],
    11: ["Mobile money / fintech transaction growth", 0.25, ""],  # +25% YoY
    12: ["Youth unemployment rate", 0.091, ""],  # 9.1%
    13: ["Government debt service / revenue ratio", 0.069, ""],  # 6.9%
    14: ["Crypto & dark-pool transaction estimate", 0.05, ""],  # 5% of GDP
    15: ["Social media toxicity / hate-speech index", 0.35, ""],
}

# ==================== 第二步：内部校准参数（无需修改） ====================
# 以下参数基于历史案例的贝叶斯校准及理论推导得出，已固化在工具中。
PARAMS = {
    # 生存与繁荣阈值
    'delta': 1.0,   # 最低生存阈值δ（标准化后基准）
    'R': 2.0,       # 社会避险基线R（标准化后基准）
    'R_plus': 2.2,  # 繁荣过剩阈值R⁺（标准化后基准）
    # 行为因子调节系数
    'alpha': 0.1,   # 正商因子避险激励系数
    'rho': 0.2,     # 心理恢复激励系数
    'mu': 0.05,     # 负商耦合抑制系数
    'kappa': 0.05,  # 负商因子过剩激励系数
    'chi': 0.1,     # 密度抑制系数
    # 网络动态系数
    'beta_plus': 0.1,   # 正连通度增长系数
    'beta_minus': 0.08, # 负连通度增长系数
    'tau': 0.15,        # 吸引力放大系数
    'iota': 0.2,        # 分裂放大系数
    'zeta_plus': 0.05,  # 正连通度耗散系数
    'zeta_minus': 0.07, # 负连通度耗散系数
    'lambda_': 0.1,     # 跨期折损率λ
    # 系统诊断系数
    'omega': 4.1,       # 负商破坏力放大系数ω
}

# ==================== 第三步：核心计算引擎（无需修改） ====================
def quick_diagnose(proxy_values, params):
    """
    快速诊断主函数。
    输入：15维代理变量值列表， 参数字典。
    输出：ϕ⁺, ϕ⁻, TP, 诊断状态。
    """
    # 1. 将输入列表转为更易读的变量名（按顺序对应）
    # 经济与信用基础
    gdp_growth, non_cash_ratio, npl_ratio, shadow_economy, gini, polarization, \
    net_migration, digital_coverage, electricity_access, internet_penetration, \
    fintech_growth, youth_unemployment, debt_service_ratio, crypto_estimate, \
    toxicity_index = proxy_values

    # 2. 计算广义能量P的代理（综合经济与数字化水平）
    P = (gdp_growth * 0.4 + digital_coverage * 0.3 + internet_penetration * 0.3) * 10  # 放大到合理量纲

    # 3. 计算关键中间变量（使用校准参数和代理变量映射）
    # 正信用编码 K⁺： 非现金支付高、不良贷款低则信用高
    K_plus = non_cash_ratio * (1 - npl_ratio*5)  # 简单线性映射，NPL影响大
    K_plus = np.clip(K_plus, 0.1, 0.95)  # 限制在合理范围
    
    # 负信用编码 K⁻： 影子经济和暗池交易占比高则负信用高
    K_minus = 0.3 * shadow_economy + 0.7 * crypto_estimate
    K_minus = np.clip(K_minus, 0.05, 0.8)
    
    # 正商因子 σ⁺： 由低极化、高净移民、低毒性支持
    sigma_plus = 0.7 - polarization*0.5 + net_migration*0.01 - toxicity_index*0.3
    sigma_plus = np.clip(sigma_plus, 0.1, 0.9)
    
    # 负商因子 σ⁻： 由青年失业、高债务、高极化驱动
    sigma_minus = 0.1 + youth_unemployment*0.6 + debt_service_ratio*0.3 + polarization*0.4
    sigma_minus = np.clip(sigma_minus, 0.05, 0.8)
    
    # 文明吸引力 A： 净移民、数字覆盖、低毒性的函数
    A = 0.3 + net_migration*0.05 + digital_coverage*0.2 - toxicity_index*0.15
    A = np.clip(A, 0.1, 0.9)
    
    # 社会分裂度 D： 直接使用极化指数
    D = polarization
    
    # 惩罚强度 Λ： 低影子经济、低不良贷款代表制度强
    Lambda = 1.5 - shadow_economy - npl_ratio*10
    Lambda = np.clip(Lambda, 0.5, 3.0)
    
    # 叙事抑制 Ψ： 低毒性指数代表社会叙事健康
    Psi = 1.0 - toxicity_index
    
    # 能量密度 G： 综合生产率，用人均GDP增长和电力接入代理
    G = (gdp_growth * 10) * 0.7 + electricity_access * 0.3
    
    # 心理恢复 H： 与青年失业和毒性负相关
    H = 0.8 - youth_unemployment*0.5 - toxicity_index*0.3
    H = np.clip(H, 0.2, 1.0)

    # 4. 调用七方程核心逻辑（简化稳态计算版，非微分方程）
    # 方程1 & 2: 计算传输量 T
    T_plus = sigma_plus * max(P - params['delta'], 0) * K_plus * np.exp(-params['lambda_'] * 1)
    T_minus = sigma_minus * max(P - params['R_plus'], 0) * K_minus * np.exp(-params['lambda_'] * 1)
    
    # 方程3 & 4: 计算σ的变化趋势（用以判断方向）
    d_sigma_plus = params['alpha'] * max(params['R'] - P, 0) + params['rho'] * H - params['mu'] * sigma_minus
    d_sigma_minus = params['kappa'] * max(P - params['R_plus'], 0) - Lambda * Psi * sigma_minus - params['chi']/G
    
    # 方程5 & 6: 计算连通度 ϕ （在假设系统处于准稳态下）
    # 假设平均传输量等于当前传输量，解出稳态 ϕ
    phi_plus = (params['beta_plus'] * T_plus * (1 + params['tau'] * A)) / params['zeta_plus'] if params['zeta_plus'] > 0 else 0
    phi_minus = (params['beta_minus'] * T_minus * (1 + params['iota'] * D)) / params['zeta_minus'] if params['zeta_minus'] > 0 else 0
    
    phi_plus = np.clip(phi_plus, 0.05, 0.8)
    phi_minus = np.clip(phi_minus, 0.02, 0.6)
    
    # 方程7: 计算跃迁潜力 TP
    # 简化计算：CCA⁺ 正比于 T_plus 积分， CCA⁻ 正比于 T_minus 积分
    CCA_plus = T_plus * 10  # 时间积分尺度因子
    CCA_minus = T_minus * 10
    eta = 1.0 - gini  # 公平效率 η 简化为 (1 - Gini)
    TP = CCA_plus * eta - params['omega'] * CCA_minus

    # 5. 根据阈值进行诊断
    status = "待诊断"
    if phi_plus >= 0.33 and phi_minus <= 0.10 and TP >= 0.52:
        status = "✅ 深度正跃迁"
    elif phi_plus >= 0.33 and phi_minus <= 0.18 and TP >= 0.15:
        status = "⚠️ 脆弱正跃迁/停滞"
    elif phi_minus > 0.18 and TP < 0.15:
        status = "🚨 负跃迁预警"
    else:
        status = "⚖️ 阈值徘徊"

    return phi_plus, phi_minus, TP, status

# ==================== 第四步：运行与输出 ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("         商理论 (Shang Theory) 快速诊断工具")
    print("="*60)
    
    # 从输入字典提取数值列表
    proxy_list = [item[1] for item in input_data.values()]
    case_name = "新加坡2024"  # 你可以修改这里为当前分析的对象名
    
    print(f"\n📊 正在诊断案例: 【{case_name}】")
    print("📥 使用的15维代理变量:")
    for key, (name, value, _) in input_data.items():
        print(f"    {key:2d}. {name:<40} : {value:>6.3f}")
    
    # 调用核心函数
    phi_plus, phi_minus, TP, status = quick_diagnose(proxy_list, PARAMS)
    
    # 打印最终诊断报告
    print("\n" + "="*60)
    print("                   诊断报告")
    print("="*60)
    print(f"核心指标 (ϕ⁺):          {phi_plus:.3f}  |  阈值 ≥ 0.33 | {'✅ 达标' if phi_plus >= 0.33 else '❌ 未达'}")
    print(f"核心指标 (ϕ⁻):          {phi_minus:.3f}  |  安全 ≤ 0.10 | {'✅ 安全' if phi_minus <= 0.10 else '⚠️ 超标'}")
    print(f"跃迁潜力 (TP):          {TP:.3f}  |  目标 ≥ 0.52 | {'✅ 充足' if TP >= 0.52 else '⚠️ 不足'}")
    print(f"系统状态:               {status}")
    print("="*60)
    
    # 提供解读
    print("\n📈 简要解读:")
    if "深度正跃迁" in status:
        print("    - 系统协作网络健康，处于积极发展轨道。")
    elif "脆弱" in status:
        print("    - 系统具有正向潜力，但基础不牢，需关注风险点。")
    elif "负跃迁预警" in status:
        print("    - 系统负向网络已高度连通，存在崩溃风险，亟需干预。")
    else:
        print("    - 系统处于临界状态，微小变化可能导致方向性转变。")
    
    # 提示关键风险/优势因子
    print(f"\n🔍 关键影响因素:")
    proxy_list = [item[1] for item in input_data.values()]
    if proxy_list[4] > 0.4:  # Gini
        print(f"    - 收入不平等(Gini指数: {proxy_list[4]:.2f})较高，压制了系统效率(η)。")
    if proxy_list[11] > 0.15:  # 青年失业
        print(f"    - 青年失业率({proxy_list[11]:.1%})是负商(σ⁻)主要滋生源。")
    if proxy_list[5] > 0.45:  # 极化
        print(f"    - 社会极化指数({proxy_list[5]:.2f})过高，严重抑制正商合作意愿(σ⁺)。")
    if proxy_list[2] < 0.03:  # NPL
        print(f"    - 不良贷款率({proxy_list[2]:.1%})较低，支持了正信用编码(K⁺)。")
    
    print("\n💡 提示：详细分析请参考完整版理论模型与案例研究。")