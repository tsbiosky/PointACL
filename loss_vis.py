import numpy as np
import open3d as o3d

path='./loss.npz'
npz = np.load(path, allow_pickle=True)
losslist = npz['loss']
losslist12=npz['loss12']
losslista1=npz['lossa1']
losslista2=npz['lossa2']
import matplotlib.pyplot as plt

x=[i for i in range(len(losslist))]
plt.plot(x, losslist,label='loss')
plt.plot(x, losslist12, label="loss_aug1&aug2")
# plotting the line 2 points
plt.plot(x, losslista2, label="loss_adv&aug2")
plt.plot(x, losslista1, label="loss_adv&aug1")

# naming the x axis
plt.xlabel('epoch')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('loss')
plt.legend(loc="center left")
# function to show the plot
plt.savefig("mygraph.png")
