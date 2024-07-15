import torch
import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def forward_hook(module, input, output):
    print(f"Inside forward hook: {module}")
    print(f"Input: {input}")    # input is a tuple
    print(f"Output: {output}")
    new_output = module.forward(input[0])  
    print(f"New output: {new_output}")
    return output


model = SimpleModel()


hook1 = model.fc1.register_forward_hook(forward_hook)


input_data = torch.randn(1, 10)
output = model(input_data)

hook1.remove()