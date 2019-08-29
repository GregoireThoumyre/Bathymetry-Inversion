#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################################################################
# Plot 4 figure with stats of prediction on 1000 cases               #
# prediction with cnn, mlp, greg method                              #
######################################################################


import context
import numpy as np
import matplotlib.pyplot as plt


rmse1 = np.load('others/rmse1.npy')
rmse2 = np.load('others/rmse2.npy')
rmse3 = np.load('others/rmse3.npy')

me1 = np.load('others/me1.npy')
me2 = np.load('others/me2.npy')
me3 = np.load('others/me3.npy')


n1 = rmse1.__len__()
n2 = rmse2.__len__()
n3 = rmse3.__len__()

p1 = 0.1
p2 = 0.1
p3 = 0.1
xrmse1 = np.arange(p1, 10+p1, p1)
xrmse2 = np.arange(p2, 10+p2, p2)
xrmse3 = np.arange(p3, 10+p3, p3)

print(xrmse1)

hrmse1 = [np.sum(rmse1<=(xrmse1[i+1])-p1)-np.sum(rmse1<(xrmse1[i]-p1)) for i in range(xrmse1.__len__()-1)]
hrmse2 = [np.sum(rmse2<=(xrmse2[i+1])-p2)-np.sum(rmse2<(xrmse2[i]-p2)) for i in range(xrmse2.__len__()-1)]
hrmse3 = [np.sum(rmse3<=(xrmse3[i+1])-p3)-np.sum(rmse3<(xrmse3[i]-p3)) for i in range(xrmse3.__len__()-1)]

xme1 = np.arange(0, 10+p1, p1)#me1.max())
xme2 = np.arange(0, 10+p2, p2)#me2.max())
xme3 = np.arange(0, 10+p3, p3)#me3.max())

hme1 = [np.sum(me1<=(xme1[i+1]-p1))-np.sum(me1<(xme1[i]-p1)) for i in range(xme1.__len__()-1)]
hme2 = [np.sum(me2<=(xme2[i+1]-p2))-np.sum(me2<(xme2[i]-p2)) for i in range(xme2.__len__()-1)]
hme3 = [np.sum(me3<=(xme3[i+1]-p3))-np.sum(me3<(xme3[i]-p3)) for i in range(xme3.__len__()-1)]


# Plot root mean square error
fig1 = plt.figure(1)
fig1.suptitle('Root Mean Square Error for each 1000 cases')
plt.subplot(3,1,1)
plt.plot(rmse1, label='Predicted with CNN from TS')
plt.legend()
plt.subplot(3,1,2)
plt.plot(rmse2, label='Predicted with MLP from celerity')
plt.ylabel('rmse')
plt.legend()
plt.subplot(3,1,3)
plt.plot(rmse3, label='Predicted with Gregs method')
plt.xlabel('predictions')
plt.legend()

# Plot mean error
fig2 = plt.figure(2)
fig2.suptitle('Mean Error for each 1000 cases')
plt.subplot(3,1,1)
plt.plot(me1, label='Predicted with CNN from TS')
plt.legend()
plt.subplot(3,1,2)
plt.plot(me2, label='Predicted with MLP from celerity')
plt.legend()
plt.subplot(3,1,3)
plt.plot(me3, label='Predicted with Gregs method')
plt.legend()

# Plot number of case with rmse
fig3 = plt.figure(3)
fig3.suptitle('Occurence number fonction of RMSE')
plt.subplot(3,1,1)
plt.bar(xrmse1[0:(xrmse1.__len__()-1)], hrmse1, 0.05, label='Predicted with CNN from TS')
plt.legend()
plt.subplot(3,1,2)
plt.bar(xrmse2[0:(xrmse2.__len__()-1)], hrmse2, 0.05, label='Predicted with MLP from celerity')
plt.ylabel('occurence number')
plt.legend()
plt.subplot(3,1,3)
plt.bar(xrmse3[0:(xrmse3.__len__()-1)], hrmse3, 0.05, label='Predicted with physical method')
plt.xlabel('rmse')

plt.legend()

fig4 = plt.figure(4)
fig4.suptitle('Occurence number fonction of ME')
plt.subplot(3,1,1)
plt.bar(xme1[0:(xme1.__len__()-1)], hme1, 0.05, label='Predicted with CNN from TS')
plt.legend()
plt.subplot(3,1,2)
plt.bar(xme2[0:(xme2.__len__()-1)], hme2, 0.05, label='Predicted with MLP from celerity')
plt.ylabel('occurence number')
plt.legend()
plt.subplot(3,1,3)
plt.bar(xme3[0:(xme3.__len__()-1)], hme3, 0.05, label='Predicted with physical method')
plt.xlabel('rmse')
plt.legend()

plt.show()

