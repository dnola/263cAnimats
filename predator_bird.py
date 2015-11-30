import math
import random

import numpy as np
import pybrain.structure.modules as psm
import pybrain.tools.shortcuts
import pygame
import pygame.gfxdraw
from  scipy.spatial.distance import cdist

from constants import *
from generic_bird import Bird

class PredatorBird(Bird):
    def __init__(self,x,y,env,weights=-1,angle=0):
        super().__init__(x,y,env,angle=angle,weights=weights)
        self.num_sight_pts = 7

    def run_network(self):
        actions = self.network.activate(self.sight_sensors+self.seen_predator)


        if self.flapped>0:
            self.flapped-=1
            actions[2]=min(actions)-.01

        action = np.argmax(actions)

        if action == 0:
            self.velr = actions[action]*.75
        elif action == 1:
            self.velr = -actions[action]*.75
        else:
            self.velr*=.85
            self.energy-=1
            self.accel = 4 + actions[action]
            self.vel*=1.1
            self.flapped=20+random.randint(-17,17)

    def eat(self,bird_id):
        if (self.energy<30 and random.random()>.8) or random.random()>.999:
            print("Bird eaten")
            self.vel*=.5
            for bird in self.env.social_birds[bird_id].get_nearby_birds():
                bird.fitness-=3
            del(self.env.social_birds[bird_id])#=self.env.breed_bird(self.env.social_birds)
            self.energy=100
            self.fitness+=1

    def see_food(self,food_coords):
        for idx,ray in enumerate(self.sight_rays):
            distances = cdist(ray,food_coords)
            distances_tree = cdist(ray,[(t.x,t.y) for t in self.env.trees])
            min = np.argmin(distances,axis=1)
            for i in range(self.num_sight_pts):
                lmin = min[i]
                tmin = np.argmin(distances_tree[i,:])

                if distances_tree[i,tmin] < 10:
                    break

                if distances[0,lmin] < 15:
                    self.eat(lmin)

                if distances[i,lmin] < 10:
                    if i==0:
                        self.eat(lmin)
                    self.sight_sensors[idx]=self.num_sight_pts-i
                    break

    def update_sight_and_contact(self):
        self.init_sight()
        self.see_food([[b.x,b.y] for b in self.env.social_birds])


    def wrap(self,x):
        return x%self.env.env_size

    def update(self):
        self.energy-=.025

        if self.energy<0:
            self.energy*=.9
            self.fitness+=self.energy*.0001
        self.move()

        self.update_sight_and_contact()

        self.run_network()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def draw(self):
        a = math.radians(self.angle)

        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 8, (150,   min(150 - int(150*self.energy/100.0),255),  min(150 - int(149*self.energy/100.0),255)))
        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 7, (150,   min(150 - int(150*self.energy/100.0),255),  min(150 - int(149*self.energy/100.0),255)))
        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 6, (150,   min(150 - int(150*self.energy/100.0),255),  min(150 - int(149*self.energy/100.0),255)))

        pygame.draw.aaline(self.env.win, DARK_RED, [int(self.x), int(self.y)], [int(self.x+12*math.cos(a)), int(self.y+12*math.sin(a))])

        for j,ray in enumerate(self.sight_rays):
            distances_tree = cdist(ray,[(t.x,t.y) for t in self.env.trees])
            for i, (x,y) in enumerate(ray):
                tmin = np.argmin(distances_tree[i,:])
                if distances_tree[i,tmin] < 10:
                    break
                if self.num_sight_pts-self.sight_sensors[j]<i:
                    break
                if self.energy>0:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, RED)
                else:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, DARK_RED)
