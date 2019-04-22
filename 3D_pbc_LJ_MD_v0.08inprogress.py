# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:38:54 2019
@author: Kel

TODO
remove globals
numba and JIT compiler to try and speed up loops
neighbor list algorithm
algo 37 & 40

ref 530
backup code to GitHub
"""

import numpy as np
import numpy.random as random
import math as math
import matplotlib.pyplot as plt
from numba import jit
import time

# matplotlib


#GLOBALS
num_atoms = 125 #was 125
box_dim = 7
time_step = 0.005
temp = 0.728
vel = 0
pos_pre = 0
pos = 0
cutoff = float(box_dim / 2) - 0.1  # should be less than half the box dimensions
ngr = 0
delg = 0
nhis = 80 # number of histograms
rho = 0.8442

g = 0

numiter = 100

evals = np.zeros(numiter)
tvals = np.zeros(numiter)
kvals = np.zeros(numiter)
etotvals = np.zeros(numiter)

# initially set forces to zero
force = np.zeros(( num_atoms, 3))
energy = 0.0

def main():
    
    global tvals
    global kvals 
    global evals
    global etotvals
    
    init()
    
    for i in range(0, numiter):
    
        tvals[i] = time_step*i
        
        #print(i, "before", energy)
        #force_calc()
        #print(i, "after", energy)
        force_calc()
    
        etot, k = integ()
        
        evals[i] = energy / num_atoms
        etotvals[i] = etot
        kvals[i] = 0.5 * (k[0] + k[1] + k[2]) / num_atoms
        
        rvals = algo7_radial()
        
    
    #print(etotvals)
    
    plt.plot(tvals, etotvals, label='E Total')
    plt.plot(tvals, kvals, label='Kinetic Energy')
    plt.plot(tvals, evals, label='Potential Energy')
    
    plt.xlabel('Time')
    plt.ylabel('Energy')
    
    plt.title('Energy Plot')
    plt.legend()
    
    plt.show()
    
    plt.plot(rvals, g, label='Distribution')
    plt.xlabel('r vals')
    plt.ylabel('g')
    
    plt.title('Distribution Plot')
    plt.legend()
    
    plt.show()
    
def algo7_radial():
    
    global g
    
    r = np.zeros(nhis)
    for i in range(0, nhis):
        
        r[i] = delg * (i + 0.5) # distance r
        vb = (( (i+1) * (i+1) * (i+1) ) - (i*i*i)) * (delg*delg*delg) #volume between bin i+1 and i?
        nid = (4/3)*(math.pi)*vb*rho #number of ideal gas particles in vb -> what is rho -> density
        g[i] = g[i] / (ngr*num_atoms*nid)
    
    return r
    

def new_force_calc():

    global num_atoms
    global pos
    global pos_pre
    global sumv
    global sumv2
    global vel
    global force
    global box_dim
    global energy
    global ngr
    
    energy = 0
    force = np.zeros(( num_atoms, 3))
    ecut = 4 * ( math.pow((1 / cutoff), 12) - math.pow((1 / cutoff), 6) )
    ngr = ngr + 1
    
        # start simple and just try to replace one loop
    for i in range(0, num_atoms - 1):
        
        int_arr = (np.round(pos / box_dim)).astype(int)
        diffvector = pos[i] - (pos - box_dim * int_arr)
        
        d2_arr = np.sum(diffvector*diffvector, axis=1)
        distance_arr= d2_arr/2
        
        # now try a boolean masking array
        #mask_arr = distance_arr < box_dim/2
        # skip radial calc for now
        
        mask_arr = d2_arr < cutoff*cutoff
        new_pos_arr = pos[mask_arr]
        
        r2i_arr = 1 / d2_arr[mask_arr]
        r6i_arr = r2i_arr * r2i_arr * r2i_arr
        ff_arr = 48 * r2i_arr * r6i_arr * (r6i_arr - 0.5)
        
        # update forces?
        
        diffvector = diffvector[mask_arr]
        ff_arr_multi = np.repeat(ff_arr[0], np.size(diffvector, axis=0)*3).reshape((np.size(diffvector, axis=0), 3))
        
        temp = ff_arr_multi * diffvector
        
        # not sure how to update the force
        
        energy = energy + 4 * np.sum(r6i_arr) * (np.sum(r6i_arr - 1)) - ecut

def force_calc():
    
    global num_atoms
    global pos
    global pos_pre
    global sumv
    global sumv2
    global vel
    global force
    global box_dim
    global energy
    global ngr
    
    energy = 0
    force = np.zeros(( num_atoms, 3))
    ecut = 4 * ( math.pow((1 / cutoff), 12) - math.pow((1 / cutoff), 6) )
    # START FORCE LOOP
    
    # not sure if this is corret -> need help writing this algorithm
    #for x in np.nditer(pos):
        #distances = np.subtract(x, pos)
        #force = np.sum(distances[distances < cutoff])
    
    ngr = ngr + 1
    
    for i in range(0, num_atoms - 1):
        
        atomx = pos[i][0]
        atomy = pos[i][1]
        atomz = pos[i][2]
        
        for j in range(i + 1, num_atoms):
            
            checkx = pos[j][0]
            checky = pos[j][1]
            checkz = pos[j][2]
            
            diffx = atomx - checkx
            diffy = atomy - checky
            diffz = atomz - checkz
            
            diffx = diffx - box_dim * int(round(diffx / box_dim))
            diffy = diffy - box_dim * int(round(diffy / box_dim))
            diffz = diffz - box_dim * int(round(diffz / box_dim))
            
            
            #d2 = diffx * diffx + diffy * diffy + diffz * diffz  -> changed this to be the sqrt
            d2 = diffx * diffx + diffy * diffy + diffz * diffz
            
            distance = math.sqrt(d2)
            
            if distance < box_dim / 2:
                
                # pause for a second and do calculations for radial function ALGO7
                ig = int(round(distance / delg)) - 1
                g[ig] = g[ig] + 2
            
            
            if d2 < cutoff*cutoff:
                
                
                r2i = 1 / d2
                r6i = r2i * r2i * r2i
                ff = 48 * r2i * r6i * (r6i - 0.5)
                # x
                force[i][0] = force[i][0] + ff * diffx
                force[j][0] = force[j][0] - ff * diffx
                
                # y
                force[i][1] = force[i][1] + ff * diffy
                force[j][1] = force[j][1] - ff * diffy
            
                # z
                force[i][2] = force[i][2] + ff * diffz
                force[j][2] = force[j][2] - ff * diffz
                
                #if force[i][0] > 10000:
                    #print("diffx", diffx)
                    #print("diffy", diffy)
                    #print("diffz", diffz)
                    
                    
                    #print(i, force[i][0], force[i][1], force[i][2])
                    
                    
                energy = energy + 4 * r6i * (r6i - 1) - ecut
                
                # work on intergration routine
                
    #temp_f = np.sum(force, axis=0)
    #print(temp_f)
    
    #return force, energy


def integ():
    
    global num_atoms
    global pos
    global pos_pre
    global vel
    global force
    global box_dim
    global energy
    global time_step
    global temp
    
    
    # Algorithm 6
    
    sumv = np.zeros(3)
    sumv2 = np.zeros(3)
    etot = 0
    
    for i in range(0, num_atoms):
        
        # Verlet Algorithm 4.2.3
        
        # grab pos of the atom in x, y, z
        atomx = pos[i][0]
        atomy = pos[i][1]
        atomz = pos[i][2]
    
        # grab pre_pos of the atom in x, y, z
        atomx_pre = pos_pre[i][0]
        atomy_pre = pos_pre[i][1]
        atomz_pre = pos_pre[i][2]
        
        # grab force of the atom in x, y, z
        force_x = force[i][0]
        force_y = force[i][1]
        force_z = force[i][2]
    
        #print(i, "force", force_x, force_y, force_z) # some force values are getting rather large
    
        # run eq 4.2.3 for each x, y, z
        xx = 2 * atomx - atomx_pre + time_step * time_step * force_x
        yy = 2 * atomy - atomy_pre + time_step * time_step * force_y
        zz = 2 * atomz - atomz_pre + time_step * time_step * force_z

        # velocity 4.2.4
        vi_x = (xx - atomx_pre) / (2 * time_step)
        vi_y = (yy - atomy_pre) / (2 * time_step)
        vi_z = (zz - atomz_pre) / (2 * time_step)
        
        #print(i, vi_x, vi_y, vi_z)
        
        v_new_arr = np.array([vi_x, vi_y, vi_z])
        v2_new_arr = np.array([vi_x * vi_x, vi_y * vi_y, vi_z * vi_z])

        #print(i, v_new_arr, v2_new_arr)

        # velocity center of mass
        sumv = sumv + v_new_arr
        #print(i)
        #print(v_new_arr)
        #print(i, 'sumv ', sumv) # last iteration this goes back to basicially 0 (e-9), but in the middle (i = 8 to i = 19) this value gets very large
        # total kinetic energy
        sumv2 = sumv2 + v2_new_arr
        #print(i, sumv2)
        
        # update positions of previous time
        pos_pre[i][0] = pos[i][0]
        pos_pre[i][1] = pos[i][1]
        pos_pre[i][2] = pos[i][2]
        #print(i, "pos", pos[i][0], pos[i][1], pos[i][2]) # some particles get very large position values, but then the first and last particles in the list suddenly has
        # to account for this, so their position values explode
        
        # update positions of current time
        pos[i][0] = xx
        pos[i][1] = yy
        pos[i][2] = zz
        
    #sumv = np.sum(sumv, axis=0)
    #sumv2 = np.sum(sumv*sumv, axis=0)
    
    # update temp val
    temp = (sumv2[0] + sumv2[1] + sumv2[2]) / (3 * num_atoms)
    #print('sumv: ', sumv)
    #print('temp: ', temp)
    # find total energy per particle
    etot = (energy + 0.5 * (sumv2[0] + sumv2[1] + sumv2[2])) / num_atoms
    #print(sumv2)
        
    return etot, sumv2
      
def init():
    
    
    global num_atoms
    global pos
    global pos_pre
    global sumv
    global sumv2
    global vel
    global ngr
    global delg
    global g
    
    # GENERATE A LATTICE IN THE BOX
    
    # assume (0,0,0) is the center of our box
    # so, spread particles out from -box_dim / 2 to box_dim / 2
  
    # make sure we are using a number of atoms that give a square lattice
    num_rows = round(num_atoms ** (1/3))
    extras = num_atoms - num_rows ** 3
    
    num_atoms = num_rows ** 3
    
    if extras != 0:
        print(extras, " atom(s) left out to leave a square lattice, using ", num_rows ** 3, " atoms")

    # find the spacing between each atom
    spacing = box_dim / num_rows
    
    # generate a list of position locations for a single row
    loc = np.linspace((-box_dim/2) + spacing, (box_dim / 2) - spacing, num_rows)

    # expand that row into x, y, and z to generate a list of points in 3-D
    #https://stackoverflow.com/questions/18253210/creating-a-numpy-array-of-3d-coordinates-from-three-1d-arrays
    pos = np.vstack(np.meshgrid(loc, loc, loc)).reshape(3, -1).T
    
    #print(pos)
    #print(len(pos)) - checked, does generate the proper number of points
    
    
    # GIVE EACH POINT A RANDOM VELOCITY VALUE
    vel = random.rand(num_atoms, 3) - 0.5

    # sum the velocity in x y z
    sumv = np.sum(vel, axis=0)
    sumv2 = np.sum(vel*vel, axis=0)
    
    
    # FIND VEL CENTER OF MASS / NUM_ATOMS
    sumv = sumv / num_atoms
    # MEAN SQUARED VELOCITY
    sumv2 = sumv2 / num_atoms
    
    # SCALE FACTOR OF VELOCITIES
    fs = np.sqrt(3 * temp / sumv2)
    
    # Set desired kinetic energy and set velocity center of mass to zero position previous time step
    vel = (vel - sumv) * fs
    #sumv = np.sum(vel, axis=0)
    #print(sumv)
    pos_pre = pos - vel * time_step
    
    
    # INIT FOR ALGO 7 radial
    
    ngr = 0
    delg = box_dim / (2 * nhis)
    
    g = np.zeros(nhis)

start = time.time()
main()
end = time.time()
print(end - start)