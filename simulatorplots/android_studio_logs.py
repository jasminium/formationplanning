import json
import sys
import os
import re
import numpy as np

# data indices

# 0 target x
# 1 target y
# 2 target z
# 3 vi x
# 4 vi y
# 5 vi z
# 6 v x
# 7 v y
# 8 v z
# 9 r x
# 10 r y
# 11 r z


def logcat_to_seconds(line):
    # time string 12:12:12.000
    p = re.compile(r'\d\d:\d\d:\d\d\.\d\d\d')
    m = p.search(line).group(0)
    t_a =  m.split('.')
    t_f = t_a[0]
    t_ms = float(t_a[1]) / 1000
    t_f_a = t_f.split(':')
    secs = float(t_f_a[0]) * 60 * 60 + float(t_f_a[1]) * 60 + float(t_f_a[2])
    secs = secs + t_ms
    return secs

def read_data(path, n_vals):

    # where k_i = 0.4
    # control = 5
    # vi_x is the sensed velocity
    f = open(path)

    data = []
    t = []
    dt = []
    loop = True

    while(loop):
        row = []
        for l in range(0, n_vals):
            line = f.readline()
            if not len(line):
                loop = False
                break
            t_s = logcat_to_seconds(line)
            val = line.split(':')[-1].strip()
            val_p = json.loads(val)
            if not isinstance(val_p, list):
                dt.append(val_p)
            else:
                row.append(val_p[0])
                row.append(val_p[1])
                row.append(val_p[2])
        #print(row)

        if loop:
            t.append(t_s)
            data.append(row)

    f.close()
    data = np.array(data)
    t = np.array(t)
    return data, t, dt
    dt = np.array(dt)
