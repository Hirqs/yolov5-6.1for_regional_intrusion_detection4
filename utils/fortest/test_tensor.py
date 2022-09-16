import torch
a=torch.tensor((530,940,3))
print(a)
print(a.shape)
b=a[[1,0,1,0]] # b是一个tensor 其中的数据是一个列表，由a中的数据[1，0，1，0]位置的元素组成，即[940, 530, 940, 530]
print('b',b)
print('哈哈')
