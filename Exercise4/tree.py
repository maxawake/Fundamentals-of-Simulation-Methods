# Fundamentals of simulation methods
# collisionless systems - tree computation of accelation
# 
# Philipp Girichidis, Nov. 2023
# 
# the code sets up a number of particles with random positions
# and builds a tree as discussed in the lecture
# then the monopoles are computed in the treewalk and accelerations
# are computed
#

import time
import numpy as np

# critical opening angle and smoothing length
theta_crit = 0.6
eps=0.01

# number of particles
Npart=2000

# class for particle
class Particle:
    def __init__(self, pos, mass):

        self.pos  = pos.copy()
        self.mass = mass
    
        # vectors for the acceleration
        acc_tree  = None
        acc_exact = None

    def print_particle(self):
        print("particle properties")
        print("pos : ", self.pos)
        print("mass: ", self.mass)


# class for the nodes
class Node:
    def __init__(self, length, center):
        '''
        init function set length and centre of the node
        '''
        
        # vectors for center and centre of mass
        self.center = center.copy()
        self.cm     = [0.0,0.0,0.0]

        self.len    = length
        self.mass   = None
        
        # this data holds 8 sub-nodes, which can contain stars
        # oragnisation left/right in x, y, z
        self.stars  = [ [[None,None], [None,None]], [[None,None], [None,None]] ]
        
        self.particle = None

    def calculate_multipole_moments(self):
        '''
        here we calculate the multipole moments
        
        the function stores the information in the nodes, so no input or returns
        '''
        
        if self.stars[0][0][0] is not None: # do we have subnodes?
            # recursively compute the moments there
            for ix in range(2):
                for iy in range(2):
                    for iz in range(2):            
                        self.stars[ix][iy][iz].calculate_multipole_moments()
                        
            # reset own values and collect them from the subtree, which we just have processed
            self.mass = 0.0
            self.cm   = [0.0,0.0,0.0]
            
            # get total mass first
            for ix in range(2):
                for iy in range(2):
                    for iz in range(2):            
                        self.mass += self.stars[ix][iy][iz].mass

            # then use total mass to get cm
            for ix in range(2):
                for iy in range(2):
                    for iz in range(2):
                        for i in range(3):
                            self.cm[i] += self.stars[ix][iy][iz].cm[i] * \
                            self.stars[ix][iy][iz].mass / self.mass
        else:
            if self.particle is not None:
                self.mass = self.particle.mass
                for i in range(3):
                    self.cm[i] = self.particle.pos[i];
            else:
                # nothing there (empty node), set mass and cm to zero
                self.mass = 0.0
                self.cm   = [0.0,0.0,0.0]

                
    def get_opening_angle(self, pos):
        '''
        get the opening angle under which my current node appears for a given position pos
        input : pos
        return: angle
        
        Do a very crude approximation for the angle, no fancy geometry. Fancy stuff is not
        needed here because we typically accept only small angles for which sin(theta) = theta
        
        Add angle_epsilon to aviod division by zero in case of identical posisions
        '''
        
        angle_epsilon = 1e-30
        r2 = 0.0
        for i in range(3):
            r2 = r2 + (self.cm[i]-pos[i])**2
        return self.len / np.sqrt(r2 + angle_epsilon)
        
    def walk_tree(self, pos):
        '''
        walks the tree and computes accelerations at a given position
        
        input : pos: position to compute accelations for
        return: acc: vector with accelerations

        NOTE: assumes that we have computed the moments, in particular the mass!
              Checks for the mass as a proxy for where tree exists.
        '''
        
        acc = [0.0,0.0,0.0]

        if self.mass > 0.0:
            
            theta = self.get_opening_angle(pos)
            
            # need to check for a small opening angle or whether the node has a particle
            if (theta < theta_crit) or (self.particle is not None):
                # compute acceleration and done
                            
                # TO BE FILLED IN
                #acc[0] = ...
                #acc[1] = ...
                #acc[2] = ...
                test=0
                
            else:
                # angle too big or empty node
                if self.stars[0][0][0] is not None:
                    # there are subnodes, so do the three walk on them
                    for ix in range(2):
                        for iy in range(2):
                            for iz in range(2):
                                acc_loc = self.stars[ix][iy][iz].walk_tree(pos)
                                for i in range(3):
                                    acc[i] += acc_loc[i]
        return acc


    def print_node(self):
        print("node properties")
        print("center:", self.center)
        
    def insert_particle(self, particle):
        # check if this node has a particle, then it is a leaf
        # if it has no particle, then it might be empty or a node with subnodes
        
        if self.particle is not None:
            # subnode has particle, create new set of 8 nodes
            # and move particle to one of them
            ctr_new = [0.0,0.0,0.0]
            for ix in range(2):
                for iy in range(2):
                    for iz in range(2):
                        ctr_new[0] = self.center[0] + 0.25 * (2.0*ix-1.0) * self.len
                        ctr_new[1] = self.center[1] + 0.25 * (2.0*iy-1.0) * self.len
                        ctr_new[2] = self.center[2] + 0.25 * (2.0*iz-1.0) * self.len
                        len_new = 0.5*self.len
                        self.stars[ix][iy][iz] = Node(length=len_new, center=ctr_new)
                        
            idx = [0,0,0]
            for i in range(3):
                if self.particle.pos[i] < self.center[i]:
                    idx[i] = 0
                else:
                    idx[i] = 1
            # move local current particle to subnode idx)
            self.stars[idx[0]][idx[1]][idx[2]].particle = self.particle

            # set own particle to None
            self.particle = None
            
            # now check the new particle and try to insert it
            idx = [0,0,0]
            for i in range(3):
                if particle.pos[i] < self.center[i]:
                    idx[i] = 0
                else:
                    idx[i] = 1
            self.stars[idx[0]][idx[1]][idx[2]].insert_particle(particle)
        else:
            # no particle there, move it into correct subnode
            idx = [0,0,0]
            for i in range(3):
                if particle.pos[i] < self.center[i]:
                    idx[i] = 0
                else:
                    idx[i] = 1
            if self.stars[idx[0]][idx[1]][idx[2]] is not None:
                self.stars[idx[0]][idx[1]][idx[2]].insert_particle(particle)
            else:
                self.particle = particle
    

# create particles with random positions
posx = np.random.uniform(size=Npart)
posy = np.random.uniform(size=Npart)
posz = np.random.uniform(size=Npart)
# and a list that holds the particles
p = []
for i in range(Npart):
    p.append(Particle(pos=[posx[i],posy[i],posz[i]], mass=1.0))

# measure time needed for tree
start_time_tree = time.time()

# create root node
root = Node(length=1.0,center=[0.5,0.5,0.5])

# insert particles into the tree
for i in range(Npart):
    root.insert_particle(p[i])   
    
root.calculate_multipole_moments()

# do the treewalk for all particles
for i in range(Npart):
    p[i].acc_tree = root.walk_tree(p[i].pos)

# done with tree computation
stop_time_tree = time.time()


# do the exact acceleration via direct summation

start_time_sum = time.time()

# TO BE FILLED IN: EXACT SUMMATION
#
# loop over particles
# p[i].acc_exact[0] = 
# p[i].acc_exact[1] = 
# p[i].acc_exact[2] = 

stop_time_sum = time.time()

print("timing")
print("tree      :", stop_time_tree - start_time_tree)
print("summation :", stop_time_sum - start_time_sum)

# compute error
#
err_sum = 0.0
# TO BE FILLED IN
#

print("error", err_sum/Npart)
print("number of computations:")
# TO BE FILLED IN
#

