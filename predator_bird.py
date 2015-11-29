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

    def eat(self,bird_id):
        if self.energy<50:
            self.vel*=.5
            self.env.social_birds[bird_id]=self.env.breed_bird(self.env.social_birds)
            self.energy=100
            self.fitness+=1


    def init_sight(self):
        a = math.radians(self.angle)
        offset = math.radians(12)

        self.sight_rays[0] = np.array([(self.wrap(self.x+13*i*math.cos(a)),self.wrap(self.y + 13*i*math.sin(a))) for i in range(self.num_sight_pts)])
        self.sight_rays[1] = np.array([(self.wrap(self.x+13*i*math.cos(a+offset)),self.wrap(self.y + 13*i*math.sin(a+offset))) for i in range(self.num_sight_pts)])
        self.sight_rays[2] = np.array([(self.wrap(self.x+13*i*math.cos(a-offset)),self.wrap(self.y + 13*i*math.sin(a-offset))) for i in range(self.num_sight_pts)])


        self.sight_sensors = [0,0,0]

    def update_sight_and_contact(self):
        self.init_sight()
        self.see_food([[b.x,b.y] for b in self.env.social_birds])


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
