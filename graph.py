import os
import numpy as np
from matplotlib import pyplot as plt


loss_list = np.zeros(shape=(30))

with open('train_loss/resnet_8_grayscale.txt', 'r') as f:
    for line in f.readlines():
        if 'Epoch' not in line:
            continue
        
        epoch = int(line.split('/')[0].split('[')[1])-1
        print(epoch)
        if epoch >= 30:
            break
        loss = float(line.split('Loss: ')[-1])
        loss_list[epoch] += loss

loss_list = loss_list / 36
print(loss_list)

plt.plot(np.arange(30)+1, loss_list)
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.ylim(2,6)
plt.savefig('grayscale_8_resnet.png')