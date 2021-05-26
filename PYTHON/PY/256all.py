import sys
import numpy as np
import linecache


T = 20001
max_lag_time = 2000
#step = 1 #1fs
file_name_out = './256a.vaf'
n_t0 = T - max_lag_time  #number of different starting points (T=8500)

#massDict = {'O':16.0,'H':1.008,'C':12.01,'N':14.01,'Ru':101.07,'S':32.06}
mass =[14.0067,12.01115,14.0067,12.01115,12.01115,12.01115,1.00797,12.01115,1.00797,1.00797,1.00797,1.00797,1.00797,12.01115,1.00797,1.00797,1.00797,1.00797,1.00797] #massDict[atom] #get mass of atom

#f=open('./HISTORY','r')
#s=f.read()

g1=[0]*2000
for l in range (9,14598,57):
    for j in range(l,l+55,3):
        data1=[]
        data2=[]
        for i in range(j,399459974,19972):
            line = linecache.getline('./HISTORY', i)
            #print(the_line)
            velx , vely, velz = line.split()
            data1.append([float(velx), float(vely), float(velz)]) #i is time index
            data2.append([float(velx), float(vely), float(velz)]) #i is time index

        G = [] #normalised and averaged over three components correlation vector
        for d in range(0,max_lag_time): # different values of lag time
            Cx = 0
            Cy = 0
            Cz = 0

            for t0 in range(0,n_t0): #different starting points t0
                Cx = Cx + float(data1[t0][0])*float(data2[t0+d][0]) #/(float(data1[t0][2])*float(data2[t0][2])) # sum( vx(t0)*vx(t0+d) )
                Cy = Cy + float(data1[t0][1])*float(data2[t0+d][1])
                Cz = Cz + float(data1[t0][2])*float(data2[t0+d][2])

            #average over different starting times
            Cx = Cx/n_t0
            Cy = Cy/n_t0
            Cz = Cz/n_t0`

            if d == 0:          #no lag   :     <A(0)B(0)>
                normx = Cx
                normy = Cy
                normz = Cz

            Cx = Cx/normx
            Cy = Cy/normy
            Cz = Cz/normz
            C = (Cx + Cy + Cz )/3.0    #3components averaged s.t. autocorrellation between -1 and +1
            G.append(C)

        #cross correlation can of course be outside the +1  <-->  -1 range and should be normalised by dividing by the max correlation
        G = G/np.amax(np.absolute(G))
        #G can be normalised without affecting whether F is normalised or not?

        g1 = g1 + G*mass[(j-l)/3]

#f.close()
cfile = open(file_name_out,"w")

N = G.size
time = [i for i in range(0,N)]

#write output file
for i in range(0,max_lag_time):
    #g1[i] = g1[i]+(G[i]*mass[j])
    t = time[i]
    g2 = g1[i]
    g3 = g1[i]/g1[0]

    out = "%20.9f%16.9f%16.9f\n" %(t,g2,g3)
    cfile.write(out)

    
cfile.close()
