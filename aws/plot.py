#coding=utf-8
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  

# 加载数据
filename = 'train_data.dat.201712201101'
fp = open(filename,'r')
data = fp.read()
fp.close()
data = json.loads(data)
print(data)

# 准备画图
data_len = len(data['acc'])

# 标题
plt.figure('准确率 && 损失值')

# 准备x轴
x = range(1, data_len+1)

# 画图
l1, = plt.plot(x, data['loss'], linestyle=':')
l2, = plt.plot(x, data['val_loss'], linestyle='-.')
l3, = plt.plot(x, data['val_acc'], linestyle='--')
l4, = plt.plot(x, data['acc'], linestyle='-')

# 图例
plt.legend(handles = [l1, l2, l3, l4], labels = ['Training loss value', 'Validation loss value', 'Validation accuracy',  'Training accuracy'], loc = 'best')

# 显示和保存图形
plt.show()
plt.savefig('result.png')
