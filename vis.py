from torchviz import make_dot
import torch
from Network import ScoreNetMLP,ScoreNetMLP1

# 创建模型实例
model = ScoreNetMLP1()

# 定义输入数据（这里的输入数据是示例，请根据你的实际情况修改）
x = torch.randn(64, 100)  # 根据你的输入维度修改

# 将模型放到 CPU 上，否则 make_dot 无法正常工作
model_cpu = model.to('cpu')
x_cpu = x.to('cpu')
t = None
# 前向传播，获取模型的计算图
y = model_cpu(x_cpu,t)
dot = make_dot(y, params=dict(model_cpu.named_parameters()))

# 保存计算图为 PDF 文件
dot.render("model_structure", format="pdf")