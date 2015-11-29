import pygame
import random
import math
import time
import pygame.gfxdraw
import numpy as np
import scipy
import pybrain as pb
import pybrain.structure as pbs
from  scipy.spatial.distance import cdist
import pybrain.tools.shortcuts
import pybrain.structure.modules as psm
import pandas

from constants import *

class Bird:
    def __init__(self,x,y,env,angle=0,weights=-1):

        self.my_id=int(random.randint(0,999999))
        self.env=env

        n = pybrain.tools.shortcuts.buildNetwork(3,5,3,hiddenclass=psm.LinearLayer, outclass=psm.SigmoidLayer, outputbias=False, recurrent=True)
        if weights!=-1:
            n._setParameters(weights)
        self.network = n

        self.fitness = 0
        self.energy = 50

        self.x=x
        self.y=y
        self.vel = 0
        self.accel=0
        self.velr = 0
        self.angle=angle

        self.sight_rays = [[],[],[]]
        self.sight_sensors = [0,0,0]
        self.num_sight_pts = 10

    def move(self):
        a = math.radians(self.angle)
        self.vel*=.8
        self.velr*=.95
        self.vel+=self.accel*.1
        self.vel = min(15,self.vel)
        self.vel = max(0.0001,self.vel)
        self.accel=self.accel-.5
        self.x = self.wrap(self.x+ self.vel * math.cos(a))
        self.y = self.wrap(self.y+ self.vel * math.sin(a))
        self.angle += self.velr

    def run_network(self):
        actions = self.network.activate(self.sight_sensors)
        action = np.argmax(actions)

        if action == 0:
            self.velr = actions[action]
        elif action == 1:
            self.velr = -actions[action]
        else:
            self.velr*=.85
            self.energy-=.1
            self.accel = 3
            self.flapped=20

    def get_nearby_birds(self):
        nearby = []
        for bird in self.env.social_birds:
            if bird.my_id==self.my_id:
                continue

            dist = ((self.x-bird.x)**2+(self.y-bird.y)**2)**.5
            if dist<100:
                nearby.append(bird)

        return nearby


    def eat(self,tree_id):
        if self.energy<70:
            self.vel*=.5
            self.env.trees[tree_id].bite()
            self.energy=100
            for bird in self.get_nearby_birds():
                self.fitness+=.5
            self.fitness+=1



    def update_sight_and_contact(self):
        a = math.radians(self.angle)
        offset = math.radians(12)

        self.sight_rays[0] = np.array([(self.wrap(self.x+13*i*math.cos(a)),self.wrap(self.y + 13*i*math.sin(a))) for i in range(self.num_sight_pts)])
        self.sight_rays[1] = np.array([(self.wrap(self.x+13*i*math.cos(a+offset)),self.wrap(self.y + 13*i*math.sin(a+offset))) for i in range(self.num_sight_pts)])
        self.sight_rays[2] = np.array([(self.wrap(self.x+13*i*math.cos(a-offset)),self.wrap(self.y + 13*i*math.sin(a-offset))) for i in range(self.num_sight_pts)])

        tree_coords = [[t.x,t.y] for t in self.env.trees]


        self.sight_sensors = [0,0,0]
        for idx,ray in enumerate(self.sight_rays):
            distances = cdist(ray,tree_coords)

            for i in range(self.num_sight_pts):
                lmin = np.argmin(distances[i,:])

                if distances[0,lmin] < 15:
                    self.eat(lmin)

                if distances[i,lmin] < 10:
                    if i==0:
                        self.eat(lmin)
                    self.sight_sensors[idx]=self.num_sight_pts-i
                    break


    def wrap(self,x):
        return x%self.env.env_size

    def update(self):
        self.energy-=.1

        if self.energy<0:
            self.energy*=.9
            self.fitness+=self.energy*.001
        self.move()

        self.update_sight_and_contact()

        self.run_network()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def draw(self):
        a = math.radians(self.angle)

        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 5, (30,   min(255 - int(255*self.energy/100.0),255),  min(255 - int(254*self.energy/100.0),255)))

        pygame.draw.aaline(self.env.win, BLACK, [int(self.x), int(self.y)], [int(self.x+7*math.cos(a)), int(self.y+7*math.sin(a))])

        for j,ray in enumerate(self.sight_rays):
            for i, (x,y) in enumerate(ray):
                if self.num_sight_pts-self.sight_sensors[j]<i:
                    break
                if self.energy>0:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, RED)
                else:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, DARK_RED)
