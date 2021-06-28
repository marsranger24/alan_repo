#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import getpass
import datetime
from datetime import date, timedelta, datetime, time
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.dates as mdates
import os
import csv
import pandas as pd

# import mplcursors

USER = getpass.getuser()


# In[ ]:


plt.rcParams['figure.figsize'] = [8, 5]
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:




isFile = os.path.isfile("/home/dell/source/alan_repo/python/test_plot.png")
print(isFile)

lis = os.listdir("/home/dell/source/alan_repo/python")
print(len(lis))

d={}
for file in os.listdir("/home/dell/source/alan_repo/python"):
    s_index = file.find('session')
    if (file[s_index:-1]+file[-1])=='session_results.csv':
        print(os.path.join("/home/dell/source/alan_repo/python",file))
        d[file[0:(s_index-1)]]=("/home/dell/source/alan_repo/python/"+file)
print(d)

for key in d:
    print(key)
    d[key] = pd.DataFrame (pd.read_csv(d[key]))
print(d)


# In[ ]:


golf_stat1=["apex","attack angle","backspin", "ball speed", 
            "carry", "club path", "club speed", "face angle", "launch angle", 
            "side spin", "side yards", "side total",  "smash factor", "spin axis", "spin rate" ]

golf_stat2=["apex","attack_angle","back_spin", "ball_speed", 
            "carry", "club_path", "club_speed", "face_angle", "launch_angle", 
            "side", "smash_factor", "spin_axis", "spin_rate" ]
y_axis = ["apex(yds)","attack angle(degrees)","backspin(rpm)", "ballspeed(mph)", 
            "carry(yds)", "club path(degress)", "club speed(mps)", "face angle(degrees)", "launch angle(degrees)", 
            "side(yds)", "smash factor", "spin axis(degrees)", "spin rate(rpm)" ]

device_num = len(d)
tm=list(d["tm"].head(0))
for key in d:
    print(key)
    if key != "tm":
        lm=list(d[key].head(0))
print(lm)        

pre = "offline_"
lm_index=[]
for stat in golf_stat2:
    word=pre+stat
    i=0
    for t in lm:
        if t.lower().find(word) != -1:
            lm_index.append(i)
        else:
            i=i+1
            
tm_index = []
for stat in golf_stat1:
    i = 0
    for t in tm:
        if t.lower().find(stat) < 2 and t.lower().find(stat) != -1:
            tm_index.append(i)
        else:
            i=i+1

print(lm_index)
print(tm_index)


# In[ ]:


mark_color = ["g^","r^","b^"]
color = ["-.g","-.r","-.b"]
for i in range(0,len(lm_index)):
    fig, ax = plt.subplots()
    k=0
    r = list(range(0,len(d['tm']["Club"])))
    plt.ylabel(y_axis[i], fontsize = 13, fontweight = 'bold')
    plt.xlabel('Shot', fontsize = 13, fontweight = 'bold')
    plt.title('20210617 FSG Test Summit: ' + golf_stat1[i],fontsize = 15, fontweight = 'bold') 
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(axis='y',which='major', linestyle='-', linewidth='1.25', color='black')
    ax.grid(axis='y',which='minor', linestyle=':', linewidth='0.75', color='black')
    for key in d:
        if key == "tm":
            index = tm_index[i]
            data = d[key][tm[index]]
            plt.stem(r, data, '-.y', markerfmt='y^', label=key)
        else:
            index = lm_index[i]
            data = d[key][lm[index]]
            plt.stem(r, data, color[k] , markerfmt=mark_color[k], label=key)
            k=k+1

    plt.xticks(r, d['tm']["Club"], rotation=90)
    plt.legend(bbox_to_anchor=(0.95, 1), prop={"size":9,'weight':'bold'})
    plt.legend()
    plt.xticks(fontsize=11, fontweight = 'bold')
    plt.yticks(fontsize=11, fontweight = 'bold')
#     plt.ylim(0,8000)
    plt.xlim((r[0]-1),r[-1]+1)
    fig.set_size_inches(20, 8)
    
    plt.savefig(golf_stat1[i]+"_plot")   


# In[ ]:


print(color[k])


# In[ ]:


print(np.shape(current_values))
print(current_values)


# In[ ]:


fig, ax = plt.subplots()
r = list(range(0,len(df4["Club"])))
plt.ylabel('Spin Axis (degrees)', fontsize = 13, fontweight = 'bold')
plt.xlabel('Shot', fontsize = 13, fontweight = 'bold')
plt.title('20210617 FSG Test Summit: Spin Axis',fontsize = 15, fontweight = 'bold') 
ax.minorticks_on()
# Customize the major grid
ax.grid(axis='y',which='major', linestyle='-', linewidth='1.25', color='black')
ax.grid(axis='y',which='minor', linestyle=':', linewidth='0.75', color='black')



plt.xticks(r, df4["Club"], rotation=90)
plt.stem(r, df1['offline_spin_rate'], '-.g', markerfmt='g^', label="042D")
plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left temperature', prop={"size":9,'weight':'bold'})
plt.legend()
plt.xticks(fontsize=11, fontweight = 'bold')
plt.yticks(fontsize=11, fontweight = 'bold')
plt.ylim(0,8000)
plt.xlim((r[0]-1),r[-1]+1)
fig.set_size_inches(20, 8)
plt.savefig("spin_axis_plot")


# In[ ]:




# time_axis = pd.read_excel(r"temp_data.xlsx", index_col=None, na_values=['NA'], usecols = "J", nrows = 8805)
# time_axis = np.array(time_axis)
# time_axis = time_axis.flatten()

# start = time(0, 0, 0)
# delta = timedelta(seconds=1)
# times = []
# n=8805

# for i in range(n):
#     # use complete datetime object, not only time
#     dt = datetime.combine(date.today(), time(0, 0)) + delta * i
#     times.append(dt)
    

# print(type(time_axis[0,0]))
# time_axis = datetime.datetime.strftime(time_axis)
# time_axis = dates.date2num(list(time_axis))
# print(np.shape(time_axis))
# print(time_axis) 
# print(type(time_axis))    
fig, ax = plt.subplots()
plt.ylabel('Spin Axis (degrees)', fontsize = 13, fontweight = 'bold')
plt.xlabel('Shot', fontsize = 13, fontweight = 'bold')
plt.title('20210617 FSG Test Summit: Spin Axis',fontsize = 15, fontweight = 'bold') 
ax.minorticks_on()
# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='1.25', color='black', markevery=1)
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.75', color='black')

# plt.plot(times, lm_1_encl_list, label="lm_1 encl_list", linewidth=1.5)
plt.stem(matrix2[:,0], matrix1[:,11], '-.g', markerfmt='g^', label="042D")
plt.stem(matrix2[:,0], matrix2[:,11], '-.b', markerfmt='b^', label="C21D")
plt.stem(matrix2[:,0], matrix3[:,11], '-.r', markerfmt='r^', label="5C2A")
plt.stem(matrix2[:,0], matrix4[:,15], '-.y', markerfmt='y^', label="TM")
plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left temperature', prop={"size":9,'weight':'bold'})
plt.legend()
plt.xticks(fontsize=11, fontweight = 'bold')
plt.yticks(fontsize=11, fontweight = 'bold')
plt.ylim(-21,21)
plt.xlim(0,36)
fig.set_size_inches(14, 8)
plt.savefig("spin_axis_plot")



# In[ ]:


times = np.arange('2020-08-11T00:00:00', '2020-08-11T02:00:00', dtype='datetime64[s]')  

solar_radiation_list = np.array(solar_radiation_list)

lm_1_batt_list = np.array(exmatrix[:,0])
lm_1_amb_in_list = np.array(exmatrix[:,1])
lm_1_encl_list = np.array(exmatrix[:,2])
lm_1_ambi_out_list = np.array(exmatrix[:,3])

# nan_1 = np.empty((1,np.size(times)-np.size(lm_1_batt_list)))
# nan_1[:] = np.NaN

# lm_1_batt_list = np.append(lm_1_batt_list, nan_1)
# lm_1_amb_in_list = np.append(lm_1_amb_in_list, nan_1)
# lm_1_encl_list = np.append(lm_1_encl_list, nan_1)
# lm_1_ambi_out_list = np.append(lm_1_ambi_out_list, nan_1)
# time_axis = pd.read_excel(r"temp_data.xlsx", index_col=None, na_values=['NA'], usecols = "J", nrows = 8805)
# time_axis = np.array(time_axis)
# time_axis = time_axis.flatten()

# start = time(0, 0, 0)
# delta = timedelta(seconds=1)
# times = []
# n=8805

# for i in range(n):
#     # use complete datetime object, not only time
#     dt = datetime.combine(date.today(), time(0, 0)) + delta * i
#     times.append(dt)
    

# print(type(time_axis[0,0]))
# time_axis = datetime.datetime.strftime(time_axis)
# time_axis = dates.date2num(list(time_axis))
# print(np.shape(time_axis))
# print(time_axis) 
# print(type(time_axis))    
fig, ax = plt.subplots()
tim_min = min(times)
tim_max = np.datetime64('2020-08-11T02:00:00', dtype='datetime64[s]')
plt.ylabel("Degrees(℃)", fontsize = 13, fontweight = 'bold')
plt.xlabel('Time', fontsize = 13, fontweight = 'bold')
plt.title('LM Outdoor Temperature Test No Power (08/13/2020)', fontsize = 13, fontweight = 'bold') 
ax.minorticks_on()
# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='1.25', color='black')
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.75', color='black')

# plt.plot(times, lm_1_encl_list, label="lm_1 encl_list", linewidth=1.5)
plt.plot(times, lm_1_batt_list[0:7200], label="batt temp", linewidth=2)
plt.plot(times, lm_1_amb_in_list[0:7200], label="ambient in temp", linewidth=2)
plt.plot(times, lm_1_encl_list[0:7200], label="back enclosure temp", linewidth=2)
plt.plot(times, lm_1_ambi_out_list[0:7200], label="ambient out temp",linewidth=2)
plt.legend(bbox_to_anchor=(0.214, 0.65), loc='upper left temperature', prop={"size":15,'weight':'bold'})
plt.xticks(fontsize=11, fontweight = 'bold')
plt.yticks(fontsize=11, fontweight = 'bold')
plt.ylim(20,70)



# ax2 = ax.twinx()
# ax2.set_ylabel('solar irradiance(W/m2)', fontsize = 13, fontweight = 'bold')  # we already handled the x-label with ax1
# ax2.plot(times, solar_radiation_list[1418:10419], color = 'black', label="theorertical solar irradiance", linewidth=3)
# plt.yticks(fontsize=11, fontweight = 'bold')

pd.plotting.register_matplotlib_converters()
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


plt.xlim(tim_min,tim_max)
# ax2.tick_params(axis='y', labelcolor=color)
# Get only the month to show in the x-axis:
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# plt.setp(ax.get_xticklabels(), rotation = 15) 
# plt.gca().xaxis.set_major_locator(mdates.HourLocator())
# plt.plot(time_axis, batt_therm_list, label="battery thermistor")
# plt.plot(time_axis, amb_in_list, label="ambient thermistor in LM")
# plt.plot(time_axis, amb_out_list, label="ambient thermistor outside LM")
# plt.plot(time_axis, rf_list, label="firmware rf readings")
# plt.plot(time_axis, batt_list, label="firmware battery readings")
# plt.plot(time_axis, zynq_list, label="firmware zynq readings")

# xlocator = time_axis.MinuteLocator(byminute=[0,15,30,45], interval = 1)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
plt.legend(bbox_to_anchor=(0.023, 0.927), loc='upper left', prop={"size":12,'weight':'bold'})
# plt.figure(figsize=(5,10))
fig.set_size_inches(14, 8)

# plt.tight_layout()
plt.savefig("lm_outdoor_temperature_test_no_power_08_11_2020")


# In[ ]:


print(time_axis[1])
print(np.shape(time_axis))


# In[ ]:


start = time(0, 0, 0)
delta = timedelta(seconds=1)
times = []
n=8870

for i in range(n):
    # use complete datetime object, not only time
    dt = datetime.combine(date.today(), time(0, 0)) + delta * i
    times.append(dt)
    
print(times)    


# In[ ]:


t = np.arange('2020-08-11T00:00:00', '2020-08-11T00:00:10', dtype='datetime64[s]')            
print(t)


# In[ ]:


print( datetime.strptime(exmatrix[x,2], '%m/%d/%Y %H:%M:%S').hour)


# In[ ]:


print(np.empty((1,3)))


# In[ ]:


print(exmatrix[25,0])
print(exmatrix[43,0])
time_begin = datetime.strptime('13:15:00', '%H:%M:%S')
s1 = (datetime.strptime(exmatrix[25,0], '%H:%M:%S').hour * 60 + datetime.strptime(exmatrix[25,0], '%H:%M:%S').minute) * 60 + datetime.strptime(exmatrix[25,0], '%H:%M:%S').second
s2 = (datetime.strptime(exmatrix[43,0], '%H:%M:%S').hour * 60 + datetime.strptime(exmatrix[43,0], '%H:%M:%S').minute) * 60 + datetime.strptime(exmatrix[43,0], '%H:%M:%S').second
s3 = (time_begin.hour * 60 + time_begin.minute) * 60 + time_begin.second
dt1 = s2-s1
dt2 = s3-s1
print(dt1)
print(dt2)


# In[ ]:


print(solar_radiation_list[7650])
print(np.shape(solar_radiation_list))


# In[ ]:


tim_min = max(times)
print(type(tim_min))


# In[ ]:


print(np.size(times)-np.size(lm_1_rf_list))


# In[ ]:


nan_1 = np.empty((1,500))
nan_2 = np.empty((1,np.size(times)-np.size(lm_1_rf_list)))
nan_3 = np.empty((1,np.size(times)-np.size(lm_1_amb_in_list)))
nan_1[:] = np.NaN
nan_2[:] = np.NaN
nan_3[:] = np.NaN
print(nan_1)


# In[ ]:


range_max_1 = max(exmatrix[-1,0])
print(range_max_1)


# In[ ]:




