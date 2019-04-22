# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:38:54 2019
@author: Micah Haack

TODO
remove globals DONE!!! SUCCESSFULLY COMPLETED YAY
numba and JIT compiler to try and speed up loops
neighbor list algorithm
algo 37 & 40

ref 530
backup code to GitHub : repository made, no spyder git integration, manual backup needed
"""

import numpy as np
import numpy.random as random
import math as math
import matplotlib.pyplot as plt
from numba import jit
import time

def main():
    
    # Initial Values
    num_atoms = 125
    box_dim = 7
    time_step = 0.005
    temp = 0.728
    cutoff = float(box_dim / 2) - 0.1 # should be < 1/2 the box dims
    nhis = 80 # number of histograms
    rho = 0.8442
    numiter = 100
    force = np.zeros(( num_atoms, 3))
    energy = 0.0
    ncells= 27 # b/c 3D system?
    
    rc = 2.5 # * sigma? these are values from the book but I don't remember
    rv = 2.7 # * sigma  what sigma was in terms for reduced units -> maybe 1?
    
    # data storage for values to graph
    tvals = np.zeros(numiter)
    kvals = np.zeros(numiter)
    evals = np.zeros(numiter)
    etotvals = np.zeros(numiter)
    
    # package variable for function input
    package = [num_atoms, box_dim, temp, time_step, nhis]
    
    # out variable for function output
    #out = [pos, pos_pre, sumv, sumv2, vel, ngr, delg, g]
    out = init(package)
    
    # UNPACK VALS FROM init()
    pos = out[0]
    pos_pre = out[1]
    #sumv = out[2]
    #sumv2 = out[3] unused? ^ v
    #vel = out[4]
    ngr = out[5]
    delg = out[6]
    g = out[7]
    
    for i in range(0, numiter):
    
        tvals[i] = time_step*i

        # PACK VALUES FOR force_calc()
        package = [num_atoms, pos, box_dim, cutoff, ngr, delg, g]

        out = force_calc(package)
        # unpack output from force_calc()
        force = out[0]
        energy = out[1]
        ngr = out[2]
        
        # PACK VALUES FOR integ()        
        package = [num_atoms, pos, pos_pre, force, energy, time_step, temp]
        
        out = integ(package)
        # unpack output from integ()
        etot = out[0]
        k = out[1]
        temp = out[2]
        
        evals[i] = energy / num_atoms
        etotvals[i] = etot
        kvals[i] = 0.5 * (k[0] + k[1] + k[2]) / num_atoms
        
        # PACK VALUES FOR algo7_radial()
        package = [g, nhis, delg, rho, num_atoms, ngr]
        out = algo7_radial(package)
        # unpack output from algo7_radial()
        r = out[0]
        g = out[1]
        
        
    # Begin plots
    
    plt.plot(tvals, etotvals, label='E Total')
    plt.plot(tvals, kvals, label='Kinetic Energy')
    plt.plot(tvals, evals, label='Potential Energy')
    
    plt.xlabel('Time')
    plt.ylabel('Energy')
    
    plt.title('Energy Plot')
    plt.legend()
    
    plt.show()
    
    plt.plot(r, g, label='Distribution')
    plt.xlabel('r vals')
    plt.ylabel('g')
    
    plt.title('Distribution Plot')
    plt.legend()
    
    plt.show()
    
    
    
def verlet_algo40(package):
    
    # unpack inputs
    box_dim = package[0]
    num_atoms = package[1]
    rc = package[2]
    ncells = package[3]
    pos = package[4]
    
    # make cell lists?
    out = cell_algo37([box_dim, num_atoms, rc, ncells, pos])
    hoc = out[0]
    ll = out[1]
    rn = out[2]
    
    # prep the nlist
    nlist = np.zeros(num_atoms)
    # do we not need to store the pos of particles? xv
    # or is this just a copy of the pos array?
    
    for i in range(1, num_atoms):
        
        # determine cell number
        icel = int(round(pos[i] / rn))
        
        #for j in range(1, ) # wait what is neigh?
    
    
    
def cell_algo37(package):
    
    box_dim = package[0]
    num_atoms = package[1]
    rc = package[2] # size of the cells
    ncells = package[3]
    pos = package[4]
    
    # prep the LL array    
    ll = np.zeros(num_atoms)
    
    hoc = 0 # this will be the head of chain array
    
    # determine size of cells rn >= rc
    rn = box_dim / int(round(box_dim/rc))
    
    # init the hoc array with zeros
    hoc = np.zeros(ncells - 1)
    
    # loop over particles
    for i in range(1, num_atoms):
        
        # determine the cell number
        icel = int(round(pos[i] / rn))
        
        # link list the hoc of cell icel
        ll[i] = hoc[icel]
        
        # make particle i the head of chain
        hoc[icel] = i
    
    out = [hoc, ll, rn]
    return out
    
    
def algo7_radial(package):
    
    g = package[0]
    nhis = package[1]
    delg = package[2]
    rho = package[3]
    num_atoms = package[4]
    ngr = package[5]
    
    r = np.zeros(nhis)
    for i in range(0, nhis):
        
        r[i] = delg * (i + 0.5) # distance r
        vb = (( (i+1) * (i+1) * (i+1) ) - (i*i*i)) * (delg*delg*delg) # volume between bin i+1 and i?
        nid = (4/3)*(math.pi)*vb*rho #number of ideal gas particles in vb -> what is rho -> density
        g[i] = g[i] / (ngr*num_atoms*nid)
    
    out = [r, g]
    
    return out

def force_calc(package):
    
    # unpack inputs
    num_atoms = package[0]
    pos = package[1]
    box_dim = package[2]    
    cutoff = package[3]
    ngr = package[4]
    delg = package[5]
    g = package[6]
    
    energy = 0
    force = np.zeros(( num_atoms, 3))
    ecut = 4 * ( math.pow((1 / cutoff), 12) - math.pow((1 / cutoff), 6) )
    # START FORCE LOOP
    
    
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
                    
                energy = energy + 4 * r6i * (r6i - 1) - ecut                
                
    out = [force, energy, ngr]
    return out
    
    #return force, energy


def integ(package):
    
    # unpack input
    num_atoms = package[0]
    pos = package[1]
    pos_pre = package[2]
    force = package[3]
    energy = package[4]
    time_step = package[5]
    temp = package[6]
    
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
        
        # run eq 4.2.3 for each x, y, z
        xx = 2 * atomx - atomx_pre + time_step * time_step * force_x
        yy = 2 * atomy - atomy_pre + time_step * time_step * force_y
        zz = 2 * atomz - atomz_pre + time_step * time_step * force_z

        # velocity 4.2.4
        vi_x = (xx - atomx_pre) / (2 * time_step)
        vi_y = (yy - atomy_pre) / (2 * time_step)
        vi_z = (zz - atomz_pre) / (2 * time_step)
        
        
        v_new_arr = np.array([vi_x, vi_y, vi_z])
        v2_new_arr = np.array([vi_x * vi_x, vi_y * vi_y, vi_z * vi_z])


        # velocity center of mass
        sumv = sumv + v_new_arr
        # total kinetic energy
        sumv2 = sumv2 + v2_new_arr
        
        # update positions of previous time
        pos_pre[i][0] = pos[i][0]
        pos_pre[i][1] = pos[i][1]
        pos_pre[i][2] = pos[i][2]
        
        # update positions of current time
        pos[i][0] = xx
        pos[i][1] = yy
        pos[i][2] = zz
        
    
    # update temp val
    temp = (sumv2[0] + sumv2[1] + sumv2[2]) / (3 * num_atoms)

    # find total energy per particle
    etot = (energy + 0.5 * (sumv2[0] + sumv2[1] + sumv2[2])) / num_atoms
    
    out = [etot, sumv2, temp]
        
    return out
      
def init(package):
    
    num_atoms = package[0]   
    box_dim = package[1]
    temp = package[2]
    time_step = package[3]
    nhis = package[4]
    
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

    pos_pre = pos - vel * time_step
    
    
    # INIT FOR ALGO 7 radial
    
    ngr = 0
    delg = box_dim / (2 * nhis)
    
    g = np.zeros(nhis)
    
    out = [pos, pos_pre, sumv, sumv2, vel, ngr, delg, g]
    return out

start = time.time()
main()
end = time.time()
print(end - start)
