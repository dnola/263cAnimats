import math
import random

import numpy as np
import pybrain.structure.modules as psm
import pybrain.tools.shortcuts
import pygame
import pygame.gfxdraw
from  scipy.spatial.distance import cdist

from constants import *


class Bird:
    def __init__(self,x,y,env,angle=0,weights=-1):

        self.my_id=int(random.randint(0,999999))
        self.env=env

        n = pybrain.tools.shortcuts.buildNetwork(4,5,3,hiddenclass=psm.LinearLayer, outclass=psm.SigmoidLayer, outputbias=False, recurrent=True)
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
        self.seen_predator = [0]
        self.num_sight_pts = 10

        self.flapped = 0

    def breed(self,other):
        e = self.env
        w1 = self.network.params
        w2 = self.network.params

        crossover = int(random.random()*len(w1))
        combined = list(w1[:crossover])+list(w2[crossover:])
        while random.random() < .1:
            combined[int(random.random()*len(w1))] = random.gauss(0,4)
        # mask = list(np.random.randint(0,1,size=len(w1)))
        # combined = [w1[i] if mask[i]==1 else w2[i] for i in range(len(w1))]

        b = self.__class__(random.random()*e.env_size,random.random()*e.env_size,e,weights=combined)
        return b

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
        actions = self.network.activate(self.sight_sensors+self.seen_predator)
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


    def see_food(self,food_coords):
        for idx,ray in enumerate(self.sight_rays):
            distances = cdist(ray,food_coords)

            for i in range(self.num_sight_pts):
                lmin = np.argmin(distances[i,:])

                if distances[0,lmin] < 15:
                    self.eat(lmin)

                if distances[i,lmin] < 10:
                    if i==0:
                        self.eat(lmin)
                    self.sight_sensors[idx]=self.num_sight_pts-i
                    break

    def init_sight(self):
        a = math.radians(self.angle)
        offset = math.radians(12)

        self.sight_rays[0] = np.array([(self.wrap(self.x+13*i*math.cos(a)),self.wrap(self.y + 13*i*math.sin(a))) for i in range(self.num_sight_pts)])
        self.sight_rays[1] = np.array([(self.wrap(self.x+13*i*math.cos(a+offset)),self.wrap(self.y + 13*i*math.sin(a+offset))) for i in range(self.num_sight_pts)])
        self.sight_rays[2] = np.array([(self.wrap(self.x+13*i*math.cos(a-offset)),self.wrap(self.y + 13*i*math.sin(a-offset))) for i in range(self.num_sight_pts)])


        self.sight_sensors = [0,0,0]

    def see_predator(self):
        pred_coords = [(p.x,p.y) for p in self.env.predator_birds]
        if self.seen_predator[0]>0:
            self.seen_predator[0]-=.025
        for idx,ray in enumerate(self.sight_rays):
            distances = cdist(ray,pred_coords)

            for i in range(self.num_sight_pts):
                lmin = np.argmin(distances[i,:])
                if distances[i,lmin] < 15:
                    if self.sight_sensors[idx]==0 or self.num_sight_pts-self.sight_sensors[idx]>=i:
                        self.sight_sensors[idx]=-(self.num_sight_pts-i)
                        self.seen_predator=[1]
                    break

    def update_sight_and_contact(self):
        self.init_sight()
        self.see_food([[t.x,t.y] for t in self.env.trees])
        self.see_predator()


    def wrap(self,x):
        return x%self.env.env_size

    def update(self):
        self.energy-=.1

        if self.energy<0:
            self.energy*=.95
            self.fitness+=self.energy*.0005
        self.move()

        self.update_sight_and_contact()

        self.run_network()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def draw(self):
        a = math.radians(self.angle)

        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 5, (30,   min(255 - int(255*self.energy/100.0),255),  min(255 - int(254*self.energy/100.0),255)))
        if self.seen_predator[0]>0:
            pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 2, DARK_RED)

        pygame.draw.aaline(self.env.win, BLACK, [int(self.x), int(self.y)], [int(self.x+7*math.cos(a)), int(self.y+7*math.sin(a))])

        for j,ray in enumerate(self.sight_rays):
            for i, (x,y) in enumerate(ray):
                if self.num_sight_pts-abs(self.sight_sensors[j])<i:
                    break
                if self.energy>0:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, RED)
                else:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, DARK_RED)
