import torch
import time,yaml,os
import numpy as np
from thop import profile # 用于计算FLOPs和参数量
from basicsr.archs import build_network


def profile_model(model, input_size=(1, 3, 64, 64), device='cuda', num_iter=100):
    """
    全面评估模型性能
    :param model: 待测模型 (nn.Module)
    :param input_size: 输入张量的形状 (B, C, H, W)，例如 (1, 3, 64, 64)
    :param device: 测试设备
    :param num_iter: 测速时的循环次数，次数越多越准
    """
    model.to(device)
    model.eval()

    # 构造 dummy input
    dummy_input = torch.randn(input_size).to(device)

    print(f"================ Testing Model: {model.__class__.__name__} ================")
    print(f"Input Shape: {input_size}")

    # -------------------------------------------------------
    # 1. 计算 参数量 (Params) 和 计算复杂度 (FLOPs)
    # -------------------------------------------------------
    # 注意：thop返回的是 MACs (Multiply–Accumulate Operations)
    # 通常 1 MAC ≈ 2 FLOPs，但在论文中大家习惯直接报 thop 的结果作为 FLOPs
    try:
        # thop.profile 会打印详细信息，可以用 verbose=False 关闭
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f"[Complexity] Params: {params / 1e6 :.4f} M")
        print(f"[Complexity] FLOPs (MACs): {macs / 1e9 :.4f} G")
    except Exception as e:
        print(f"[Error] FLOPs calculation failed: {e}")

    # -------------------------------------------------------
    # 2. 测试 显存占用 (Peak Memory)
    # -------------------------------------------------------
    # 必须先清理缓存并重置统计数据
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    # 获取最大显存占用
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # 转换为 MB
    print(f"[Memory] Peak Memory: {mem:.2f} MB")

    # -------------------------------------------------------
    # 3. 测试 推理速度 (Inference Speed)
    # -------------------------------------------------------
    # 3.1 预热 (Warmup) - 关键步骤！
    # GPU 需要时间把 CUDA kernel 加载进来，不做预热测出来会很慢
    print("Warming up...", end="")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    print(" Done.")

    # 3.2 正式测速 (使用 torch.cuda.Event 进行精确计时)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_iter, 1))
    
    print(f"Measuring latency over {num_iter} runs...")
    with torch.no_grad():
        for rep in range(num_iter):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            
            # 等待 GPU 完成所有任务
            torch.cuda.synchronize() 
            curr_time = starter.elapsed_time(ender) # 返回的是毫秒
            timings[rep] = curr_time

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    fps = 1000 / mean_time

    print(f"[Speed] Average Latency: {mean_time:.4f} ms")
    print(f"[Speed] FPS: {fps:.2f}")
    print("=================================================================\n")
    
    return {
        "params_M": params / 1e6,
        "flops_G": macs / 1e9,
        "memory_MB": mem,
        "latency_ms": mean_time,
        "fps": fps
    }

# ================= 使用示例 =================

# 假设你有两个模型定义 (这里用简单的卷积模拟)
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv(x)

import torch
import time
import numpy as np
from thop import profile # 用于计算FLOPs和参数量

def profile_model(model, input_size=(1, 3, 64, 64), device='cuda', num_iter=100):
    """
    全面评估模型性能
    :param model: 待测模型 (nn.Module)
    :param input_size: 输入张量的形状 (B, C, H, W)，例如 (1, 3, 64, 64)
    :param device: 测试设备
    :param num_iter: 测速时的循环次数，次数越多越准
    """
    model.to(device)
    model.eval()

    # 构造 dummy input
    dummy_input = torch.randn(input_size).to(device)

    print(f"================ Testing Model: {model.__class__.__name__} ================")
    print(f"Input Shape: {input_size}")

    # -------------------------------------------------------
    # 1. 计算 参数量 (Params) 和 计算复杂度 (FLOPs)
    # -------------------------------------------------------
    # 注意：thop返回的是 MACs (Multiply–Accumulate Operations)
    # 通常 1 MAC ≈ 2 FLOPs，但在论文中大家习惯直接报 thop 的结果作为 FLOPs
    try:
        # thop.profile 会打印详细信息，可以用 verbose=False 关闭
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f"[Complexity] Params: {params / 1e6 :.4f} M")
        print(f"[Complexity] FLOPs (MACs): {macs / 1e9 :.4f} G")
    except Exception as e:
        print(f"[Error] FLOPs calculation failed: {e}")

    # -------------------------------------------------------
    # 2. 测试 显存占用 (Peak Memory)
    # -------------------------------------------------------
    # 必须先清理缓存并重置统计数据
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    # 获取最大显存占用
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # 转换为 MB
    print(f"[Memory] Peak Memory: {mem:.2f} MB")

    # -------------------------------------------------------
    # 3. 测试 推理速度 (Inference Speed)
    # -------------------------------------------------------
    # 3.1 预热 (Warmup) - 关键步骤！
    # GPU 需要时间把 CUDA kernel 加载进来，不做预热测出来会很慢
    print("Warming up...", end="")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    print(" Done.")

    # 3.2 正式测速 (使用 torch.cuda.Event 进行精确计时)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_iter, 1))
    
    print(f"Measuring latency over {num_iter} runs...")
    with torch.no_grad():
        for rep in range(num_iter):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            
            # 等待 GPU 完成所有任务
            torch.cuda.synchronize() 
            curr_time = starter.elapsed_time(ender) # 返回的是毫秒
            timings[rep] = curr_time

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    fps = 1000 / mean_time

    print(f"[Speed] Average Latency: {mean_time:.4f} ms")
    print(f"[Speed] FPS: {fps:.2f}")
    print("=================================================================\n")
    
    return {
        "params_M": params / 1e6,
        "flops_G": macs / 1e9,
        "memory_MB": mem,
        "latency_ms": mean_time,
        "fps": fps
    }

# ================= 使用示例 =================

# 假设你有两个模型定义 (这里用简单的卷积模拟)
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv(x)

import torch
import time
import numpy as np
from thop import profile # 用于计算FLOPs和参数量


import torch.nn as nn

class WrappedSRNet(nn.Module):
    """
    把 net_g(input, sigma, cond_lq) 包一层，变成只接收一个张量输入的模型：
      forward(x) 实际调用 net_g(x, sigma, cond_lq)
    """
    def __init__(self, net_g, sigma, cond_lq):
        super().__init__()
        self.net_g = net_g
        # sigma: 标量或 shape=[B] 的张量
        # cond_lq: 低分图 or 上采样后的 LQ，shape=[B,C,H_lq,W_lq] 或 [B,C,H_hr,W_hr]
        self.sigma = sigma
        self.cond_lq = cond_lq

    def forward(self, x):
        # x 相当于 lq_up + sigma * noise，是 HR 分辨率的输入
        return self.net_g(x, self.sigma, self.cond_lq)

def profile_model(model, input_size=(1, 3, 64, 64), device='cuda', num_iter=100):
    """
    全面评估模型性能
    :param model: 待测模型 (nn.Module)
    :param input_size: 输入张量的形状 (B, C, H, W)，例如 (1, 3, 64, 64)
    :param device: 测试设备
    :param num_iter: 测速时的循环次数，次数越多越准
    """
    model.to(device)
    model.eval()

    # 构造 dummy input
    dummy_input = torch.randn(input_size).to(device)

    print(f"================ Testing Model: {model.__class__.__name__} ================")
    print(f"Input Shape: {input_size}")

    # -------------------------------------------------------
    # 1. 计算 参数量 (Params) 和 计算复杂度 (FLOPs)
    # -------------------------------------------------------
    # 注意：thop返回的是 MACs (Multiply–Accumulate Operations)
    # 通常 1 MAC ≈ 2 FLOPs，但在论文中大家习惯直接报 thop 的结果作为 FLOPs
    try:
        # thop.profile 会打印详细信息，可以用 verbose=False 关闭
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        print(f"[Complexity] Params: {params / 1e6 :.4f} M")
        print(f"[Complexity] FLOPs (MACs): {macs / 1e9 :.4f} G")
    except Exception as e:
        print(f"[Error] FLOPs calculation failed: {e}")

    # -------------------------------------------------------
    # 2. 测试 显存占用 (Peak Memory)
    # -------------------------------------------------------
    # 必须先清理缓存并重置统计数据
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    # 获取最大显存占用
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024 # 转换为 MB
    print(f"[Memory] Peak Memory: {mem:.2f} MB")

    # -------------------------------------------------------
    # 3. 测试 推理速度 (Inference Speed)
    # -------------------------------------------------------
    # 3.1 预热 (Warmup) - 关键步骤！
    # GPU 需要时间把 CUDA kernel 加载进来，不做预热测出来会很慢
    print("Warming up...", end="")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    print(" Done.")

    # 3.2 正式测速 (使用 torch.cuda.Event 进行精确计时)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_iter, 1))
    
    print(f"Measuring latency over {num_iter} runs...")
    with torch.no_grad():
        for rep in range(num_iter):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            
            # 等待 GPU 完成所有任务
            torch.cuda.synchronize() 
            curr_time = starter.elapsed_time(ender) # 返回的是毫秒
            timings[rep] = curr_time

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    fps = 1000 / mean_time

    print(f"[Speed] Average Latency: {mean_time:.4f} ms")
    print(f"[Speed] FPS: {fps:.2f}")
    print("=================================================================\n")
    
    return {
        "params_M": params / 1e6,
        "flops_G": macs / 1e9,
        "memory_MB": mem,
        "latency_ms": mean_time,
        "fps": fps
    }

# ================= 使用示例 =================

# 假设你有两个模型定义 (这里用简单的卷积模拟)
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    import yaml
    from basicsr.archs import build_network
    import torch.nn.functional as F

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 读取 test 配置，拿到 network_g 和 sigma_max 等
    opt_path = "options/test/ctmsr_test.yml"  # 按你的实际路径
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)

    net_opt = opt["network_g"]
    sf = int(net_opt.get("scale", 4))
    use_enc = bool(net_opt.get("use_enc", False))

    train_opt = opt["train"]
    if train_opt.get("consistency_opt"):
        diffusion_opt = train_opt["consistency_opt"]
    elif train_opt.get("diffloss_opt"):
        diffusion_opt = train_opt["diffloss_opt"]
    else:
        raise ValueError("No consistency_opt or diffloss_opt in train options.")
    sigma_max = float(diffusion_opt["sigma_max"])

    # 2. 构建 net_g（不走 Model 壳，直接网络本体）
    net_g = build_network(net_opt).to(device)
    net_g.eval()

    # 3. 构造一个与真实推理一致的 dummy 输入
    #    假设 LR 为 64x64，scale=4 -> HR 256x256
    B, C, H_lq, W_lq = 1, 3, 128, 128
    lq = torch.randn(B, C, H_lq, W_lq, device=device)  # 只是 dummy，用于复杂度测试

    # 和 EdmUNetRealModel.test 一致的上采样 + 随机噪声
    lq_up = F.interpolate(lq, scale_factor=sf, mode="bicubic")
    sigma = torch.as_tensor(sigma_max, device=device)
    latent = torch.randn_like(lq_up, device=device)
    input_tensor = lq_up + sigma * latent  # 这个就是 net_g 的第一个输入

    cond_lq = lq if use_enc else lq_up

    # 4. 用 WrappedSRNet 把多输入的 net_g 包装成单输入模型
    wrapped_model = WrappedSRNet(net_g, sigma, cond_lq)

    # 5. 调用 profile_model
    #    注意：input_size 要和 input_tensor 的 shape 对齐
    input_size = input_tensor.shape  # (B,3,H_hr,W_hr)
    results = profile_model(wrapped_model, input_size=input_size, device=device, num_iter=100)
    print(results)