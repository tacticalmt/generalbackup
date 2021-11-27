import os
import shutil
import numpy as np

root = os.path.abspath('../../dataset/AwA2-features/Animals_with_Attributes2/Features/ResNet101/')
a_path = os.path.join(root, 'AwA2-features.txt')
word_att = []
att = open(a_path, 'r')  # 打开attribute文件
while True:
    temp_att = att.readline().strip()
    if not temp_att:
        break
    word_att.append(temp_att)
att.close()
divided_att = []
for key_word in word_att:
    v = key_word.split(' ')
    divided_att.append(v)
divided_np = np.array(divided_att)
# att_int = divided_np.astype(int)
print(divided_np.shape)
mem = divided_np.shape
