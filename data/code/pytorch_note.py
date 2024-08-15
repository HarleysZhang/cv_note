####################卷积神经网络计算量/参数量分析工具#####################
import torchvision, torch

model = torchvision.models.resnet50()

# 1, pytorch 自带输出
print(model)

# 2, torchinfo 工具
from torchinfo import summary
summary(model, (1, 3, 224, 224), depth=3) # resnet50: 25.557M 4.09G

# 3, thop 工具
from thop import profile, clever_format
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print("The resnet50 model info: ", macs, params) # resnet50: 4.134G 25.557M