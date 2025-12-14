"""
商理论 (Shang Theory) 3.2 - 核心方程实现
本模块实现了双渗流文明动态理论的七个核心动力学方程。
所有函数均设计为向量化运算，支持输入时间序列数据进行批量计算。
"""

import numpy as np

def calculate_T_plus(sigma_plus, P, delta, K_plus, lambda_, delta_t):
    """
    计算正向能量包传输 T⁺
    方程: T⁺ = σ⁺ · max(P - δ, 0) · K⁺ · exp(-λΔt)
    
    参数:
    sigma_plus (float or np.ndarray): 正商因子
    P (float or np.ndarray): 主体广义能量
    delta (float): 最低生存阈值
    K_plus (float or np.ndarray): 正信用编码强度
    lambda_ (float): 折损率系数
    delta_t (float or np.ndarray): 承诺跨期长度
    
    返回:
    float or np.ndarray: 正向传输量 T⁺
    """
    surplus = np.maximum(P - delta, 0)
    discount = np.exp(-lambda_ * delta_t)
    T_plus = sigma_plus * surplus * K_plus * discount
    return T_plus

def calculate_T_minus(sigma_minus, P, R_plus, K_minus, lambda_, delta_t):
    """
    计算负向能量包传输 T⁻
    方程: T⁻ = σ⁻ · max(P - R⁺, 0) · K⁻ · exp(-λΔt)
    
    参数:
    sigma_minus (float or np.ndarray): 负商因子
    P (float or np.ndarray): 主体广义能量
    R_plus (float): 繁荣过剩阈值
    K_minus (float or np.ndarray): 负信用编码强度
    lambda_ (float): 折损率系数
    delta_t (float or np.ndarray): 承诺跨期长度
    
    返回:
    float or np.ndarray: 负向传输量 T⁻
    """
    excess = np.maximum(P - R_plus, 0)
    discount = np.exp(-lambda_ * delta_t)
    T_minus = sigma_minus * excess * K_minus * discount
    return T_minus

def d_sigma_plus_dt(alpha, R, P, rho, H, mu, sigma_minus):
    """
    计算正商因子随时间的变化率 dσ⁺/dt
    方程: dσ⁺/dt = α·max(R - P, 0) + ρH - μσ⁻
    
    参数:
    alpha (float): 调节系数
    R (float or np.ndarray): 社会避险基线
    P (float or np.ndarray): 主体广义能量
    rho (float): 心理恢复系数
    H (float or np.ndarray): 心理恢复指数
    mu (float): 负商耦合衰减系数
    sigma_minus (float or np.ndarray): 负商因子
    
    返回:
    float or np.ndarray: 正商因子变化率
    """
    risk_pressure = np.maximum(R - P, 0)
    change = alpha * risk_pressure + rho * H - mu * sigma_minus
    return change

def d_sigma_minus_dt(kappa, P, R_plus, Lambda, Psi, sigma_minus, chi, G):
    """
    计算负商因子随时间的变化率 dσ⁻/dt
    方程: dσ⁻/dt = κ·max(P - R⁺, 0) - Λ·Ψ·σ⁻ - χ/G
    
    参数:
    kappa (float): 调节系数
    P (float or np.ndarray): 主体广义能量
    R_plus (float): 繁荣过剩阈值
    Lambda (float or np.ndarray): 惩罚强度
    Psi (float or np.ndarray): 叙事抑制因子
    sigma_minus (float or np.ndarray): 负商因子
    chi (float): 密度抑制系数
    G (float or np.ndarray): 能量密度（平均生产率）
    
    返回:
    float or np.ndarray: 负商因子变化率
    """
    excess = np.maximum(P - R_plus, 0)
    suppression = Lambda * Psi * sigma_minus
    density_inhibition = chi / G
    change = kappa * excess - suppression - density_inhibition
    return change

def d_phi_plus_dt(beta_plus, T_plus_avg, tau, A, zeta_plus, phi_plus):
    """
    计算正商网络连通度变化率 dϕ⁺/dt
    方程: dϕ⁺/dt = β⁺·⟨T⁺⟩·(1 + τA) - ζ⁺ϕ⁺
    
    参数:
    beta_plus (float): 增长系数
    T_plus_avg (float or np.ndarray): 平均正向传输量
    tau (float): 吸引力放大系数
    A (float or np.ndarray): 文明吸引力（软实力系数）
    zeta_plus (float): 耗散系数
    phi_plus (float or np.ndarray): 当前正连通度
    
    返回:
    float or np.ndarray: 正连通度变化率
    """
    growth = beta_plus * T_plus_avg * (1 + tau * A)
    dissipation = zeta_plus * phi_plus
    change = growth - dissipation
    return change

def d_phi_minus_dt(beta_minus, T_minus_avg, iota, D, zeta_minus, phi_minus):
    """
    计算负商网络连通度变化率 dϕ⁻/dt
    方程: dϕ⁻/dt = β⁻·⟨T⁻⟩·(1 + ιD) - ζ⁻ϕ⁻
    
    参数:
    beta_minus (float): 增长系数
    T_minus_avg (float or np.ndarray): 平均负向传输量
    iota (float): 分裂放大系数
    D (float or np.ndarray): 社会分裂度
    zeta_minus (float): 耗散系数
    phi_minus (float or np.ndarray): 当前负连通度
    
    返回:
    float or np.ndarray: 负连通度变化率
    """
    growth = beta_minus * T_minus_avg * (1 + iota * D)
    dissipation = zeta_minus * phi_minus
    change = growth - dissipation
    return change

def calculate_TP(CCA_plus, G, CCA_minus, omega=4.1):
    """
    计算系统跃迁潜力 TP
    方程: TP = CCA⁺ · η(G) - ω · CCA⁻
    其中 η(G) 简化为 (1 - Gini) 或根据能量密度调整的公平效率
    
    参数:
    CCA_plus (float or np.ndarray): 正商系统活性
    G (float or np.ndarray): 能量密度（用于计算η）
    CCA_minus (float or np.ndarray): 负商系统活性
    omega (float): 破坏放大系数，默认为4.1
    
    返回:
    float or np.ndarray: 跃迁潜力 TP 值
    """
    # 这里是一个简单的 η(G) 代理实现，你可以根据理论完善它
    # 例如: eta = 1.0 / (1.0 + np.exp(-G)) 或 eta = np.log(G + 1)
    eta = np.log(G + 1)  # 示例函数，请替换为你的正式定义
    TP = CCA_plus * eta - omega * CCA_minus
    return TP

if __name__ == "__main__":
    # 简单的测试代码，验证函数是否可以正常运行
    print("核心方程模块导入成功。")
    print("可调用函数: calculate_T_plus, calculate_T_minus, d_sigma_plus_dt, d_sigma_minus_dt, d_phi_plus_dt, d_phi_minus_dt, calculate_TP")
    