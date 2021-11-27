import torch.nn as nn
import torch

conv1 = nn.Linear(2, 3)
input_l = torch.randn(3, 2)
print(input_l)
out_res = conv1(input_l)
print(out_res)
input_l=input_l.cuda()
act1 = nn.ReLU(inplace=True)
#act1.cuda()
after_res = act1(1 - input_l)
after_res += act1(1 + input_l)
out=after_res
print(out)