import torch
import torch.nn as nn

x = torch.randn(1, 128)

dropout = nn.Dropout(p=0.5)


# During Training

dropout.train()
output_train = dropout(x)

print(output_train)

# During Evaluation

dropout.eval()
output_eval = dropout(x)

print(output_eval)

x1 = torch.randn(1, 1024)

fc1 = nn.Linear(1024, 128)
fc2 = nn.Linear(128, 10)

relu = nn.ReLU()

x1 = fc1(x1)
x1 = relu(x1)
x1 = dropout(x1)

x1 = fc2(x1)
print(x1.shape)