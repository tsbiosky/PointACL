#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
#plt.style.use('_mpl-gallery')

# make data
np.random.seed(1)
x = [1,3,5,8,10]
sa= [0.8071,0.7953,0.7867,0.7833,0.7763]
ra =[0.2751,0.2785,0.2867,0.2971,0.3134]
#y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
#y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

# plot
fig=plt.figure()
ax1=plt.subplot(1,2,1)
plt.plot(x, sa, linewidth=2,marker='o',linestyle='dashed')
plt.axis([0,11,0.75,0.83])
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.grid()
plt.xlabel('alpha',fontsize=20)
plt.title('Standard Accuracy(SA)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax2=plt.subplot(1,2,2)
plt.plot(x, ra, color='red',linewidth=2,marker='o',linestyle='dashed')
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.axis([0,11,0.25,0.32])
plt.grid()
plt.xlabel('alpha',fontsize=20)
plt.title('Robust Accuracy(SA)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
'''
x=[1,3,5,7,10,15,20,25,30,35]
rocl=[0.2811,0.1227,0.0956,0.08225,0.07617,0.06992,0.06419,0.062,0.0612,0.0607]
acl=[0.299,0.141,0.108,0.0911,0.0810,0.0721,0.0680,0.0656,0.06321,0.06199]
pointacl=[0.42,0.32,0.29,0.275,0.2576,0.248,0.2411,0.237,0.235,0.231]
fig=plt.figure()
ax1=plt.subplot(1,2,1)
l1,=plt.plot(x, rocl, linewidth=2,marker='o',linestyle='-',color='b')
l2,=plt.plot(x, acl, linewidth=2,marker='v',linestyle='-',color='g')
l3,=plt.plot(x, pointacl, linewidth=2,marker='s',linestyle='-',color='r')
plt.axis([0,35,0,0.5])
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.grid()
ax1.legend(handles=[l1,l2,l3],labels=['ROCL', 'ACL','PointACL(ours)'],loc='upper right',fontsize=25)
plt.xlabel('number of iteration',fontsize=20)
plt.title('RA in different iterations when budget=0.01m',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
ax2=plt.subplot(1,2,2)
rocl=np.load('./BYOL/plot/ROCL.npz')['acc']
acl=np.load('./BYOL/plot/ACL.npz')['acc']
pointacl=np.load('./BYOL/plot/PointACL.npz')['acc']
for i in range(20):
    rocl[i]=rocl[i]/100
    acl[i] = acl[i] / 100
    pointacl[i] = pointacl[i] / 100
x=[0.001*(i+1) for i in range(20)]
print(x,rocl)
l1,=plt.plot(x, rocl, linewidth=2,marker='o',linestyle='-',color='b')
l2,=plt.plot(x, acl, linewidth=2,marker='v',linestyle='-',color='g')
l3,=plt.plot(x, pointacl, linewidth=2,marker='s',linestyle='-',color='r')
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
plt.axis([0,0.02,0,0.8])
plt.grid()
ax2.legend(handles=[l1,l2,l3],labels=['ROCL', 'ACL','PointACL(ours)'],loc='upper right',fontsize=25)
plt.xlabel('Budget',fontsize=20)
plt.title('RA in different budget when iteration = 5',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
plt.show()
'''