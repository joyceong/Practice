# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
from matplotlib import animation
import pandas as pd
from time import time

trace = pd.read_csv('traceData.csv',index_col=False)
# print trace
# print len(vehicle_unique)

#for vehicle in vehicle_unique:
#    x = trace[trace['vehicle-id']==vehicle]['x-coordinate']
# x = trace.groupby(trace['vehicle-id']).sort_values(by=['time'])

trace = trace.sort_values(by=['vehicle-id','time'])
vehicle_unique = trace['vehicle-id'].unique()
'''
# plot the trace of 5415 cars
color_set = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
start = time()
#for i in range(len(vehicle_unique)):
for i in range(1000):
    x = trace[trace['vehicle-id']==vehicle_unique[i]]['x-coordinate']
    y = trace[trace['vehicle-id']==vehicle_unique[i]]['y-coordinate']
    plt.scatter(x,y, color = color_set[i%len(color_set)])
stop = time() 
print 'time:'+str(stop-start)+'s'
'''
vehicle_start = trace.groupby(by='vehicle-id').first()[['x-coordinate','y-coordinate']]
vehicle_start.rename(columns={'x-coordinate':'start-x', 'y-coordinate':'start-y'}, inplace = True)
# print vehicle_start
trace_extent = trace.join(vehicle_start,on='vehicle-id')
trace_extent['distance'] = ((trace_extent['x-coordinate']-trace_extent['start-x'])**2+(trace_extent['y-coordinate']-trace_extent['start-y'])**2)**0.5
#trace_extent['route'][0] = 0
trace_extent['route'] = ((trace_extent['x-coordinate']-trace_extent['x-coordinate'].shift(1))**2+(trace_extent['y-coordinate']-trace_extent['y-coordinate'].shift(1))**2)**0.5
print trace_extent
'''
for i in range(len(vehicle_unique)):
# for i in range(10):
    #plt.figure(i)
    plt.plot(trace_extent[trace_extent['vehicle-id']==vehicle_unique[i]]['time'],trace_extent[trace_extent['vehicle-id']==vehicle_unique[i]]['distance'])
    plt.xlabel('time')
    plt.ylabel('distance')
'''
'''
# print x.type
# print np.linspace(0,2*np.pi,128)
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)

def init():
    ax.set_xlim(521000,522000)
    ax.set_ylim(54000,56000)
    return ln,


def update(x,y):
    xdata.append(x)
    ydata.append(y)
    ln.set_data(xdata, ydata)
    return ln,

# ani = FuncAnimation(fig, update, frames=x, init_func=init, blit=True)
'''
plt.show()

