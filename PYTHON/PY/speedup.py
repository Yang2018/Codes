import linecache

from multiprocessing import Manager, Process
import time

import numpy as np



T = 20001
max_lag_time = 2000

n_t0 = T - max_lag_time

START_LINE = 9

START_LINE_NUM = 4864


MAX_LINE = 399459974

STEP_LINE = 19972

file_name = 'HISTORY'


mass = [
    14.0067, 12.01115, 14.0067, 12.01115, 12.01115, 12.01115, 1.00797, 12.01115, 1.00797,
    1.00797,  1.00797, 1.00797,  1.00797, 12.01115,  1.00797, 1.00797,  1.00797, 1.00797, 1.00797
]

data = [[] for _ in range(START_LINE_NUM)]


def func_calc(pid, start, end, data, manager_dict):
    g_list = []
    for i in range(start, end):
        G = []

        for d in range(0, max_lag_time):
            Cx = 0
            Cy = 0
            Cz = 0

            for t0 in range(0, n_t0):
                Cx += data[i][t0][0] * data[i][t0 + d][0]
                Cy += data[i][t0][1] * data[i][t0 + d][1]
                Cz += data[i][t0][2] * data[i][t0 + d][2]


            Cx /= n_t0
            Cy /= n_t0
            Cz /= n_t0

            if d == 0:
                normx = Cx
                normy = Cy
                normz = Cz

            Cx /= normx
            Cy /= normy
            Cz /= normz

            C = (Cx + Cy + Cz) / 3.0
            G.append(C)

        g_list.append(G)

    manager_dict[pid] = g_list


def calculate():

    for i in range(START_LINE_NUM):
        for j in range(START_LINE, MAX_LINE, STEP_LINE):
            line_number = i * 3 + j
            line = linecache.getline(file_name, line_number)

            velx, vely, velz = line.split()

            x = float(velx)
            y = float(vely)
            z = float(velz)

            data[i].append([x, y, z])



    manager = Manager()
    manager_dict = manager.dict()
    process_num = 38
    args = []

    for k in range(0,process_num):
        args.append((k,k*128,k*128+128,data,manager_dict))



    job_list = []
    for i in range(process_num):

        p = Process(target=func_calc, args=args[i])
        job_list.append(p)
        p.start()

    for t in job_list:
        t.join()

    g_list = []

    for m in range(process_num):
        g_list.extend(manager_dict[m])

    g1 = [0] * max_lag_time  
    for i, G in enumerate(g_list):

        G = G / np.amax(np.absolute(G))

        g1 += G * mass[i % 19]


    return g1


def save(g1):
    file_name_out = './256a.vaf'
    cfile = open(file_name_out, "w")


    N = max_lag_time
    time = [t for t in range(0, N)]

    # write output file
    for k in range(0, max_lag_time):

        t = time[k]
        g2 = g1[k]
        g3 = g1[k] / g1[0]

        out = "%20.9f%16.9f%16.9f\n" % (t, g2, g3)
        cfile.write(out)

    cfile.close()


if __name__ == '__main__':
    from multiprocessing import cpu_count
    print(cpu_count())

    start_time = time.time()
    g1 = calculate()
    end_time = time.time()
    print(end_time - start_time)
    save = save(g1)