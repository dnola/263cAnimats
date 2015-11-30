import math
import random

import numpy as np
import pybrain.structure.modules as psm
import pybrain.tools.shortcuts
import pygame
import pygame.gfxdraw

from constants import *
from generic_bird import Bird


class SocialBird(Bird):
    def __init__(self,x,y,env,weights=-1,angle=0,chirp_df=3):

        super().__init__(x,y,env=env,angle=angle)

        # n = pybrain.tools.shortcuts.buildNetwork(8,10,5,5,hiddenclass=psm.LinearLayer, outclass=psm.SigmoidLayer, outputbias=False, recurrent=True)
        n = pybrain.tools.shortcuts.buildNetwork(7+chirp_df,20,3+chirp_df,hiddenclass=psm.LSTMLayer, outclass=psm.SigmoidLayer, outputbias=True, recurrent=True)
        # n = pybrain.tools.shortcuts.buildNetwork(7+chirp_df,12,8,3+chirp_df,hiddenclass=psm.LinearLayer, outclass=psm.SigmoidLayer, outputbias=True, recurrent=True)

        if weights!=-1:
            n._setParameters(weights)
        self.network = n

        self.reponse_time_series = []
        self.correlation = []

        self.sound_origin = [x,y]
        self.sound_sensors = [0 for i in range(chirp_df)]
        self.sound_direction = [0,0]
        self.chirp_df=chirp_df

        self.facing_vector = [0,0]
        self.sound_timer=0
        self.pattern_heard = [0 for i in range(chirp_df)]


        self.chirped = 0



    def chirp(self,sn):
        for bird in self.get_nearby_birds():
            bird.heard_sound(self.x,self.y,sn)

    def heard_sound(self,sx,sy,sn):
        self.sound_origin = [sx,sy]
        self.pattern_heard = sn
        self.sound_timer=300

    def run_network(self):
        actions = self.network.activate(list(self.sight_sensors)+self.seen_predator+list(self.sound_sensors)+list(self.pattern_heard)+[self.energy])

        if random.random()>.9 or len(self.reponse_time_series) < 3:
            to_add = list(list(self.sight_sensors)+self.seen_predator+list(self.sound_sensors)+list(self.pattern_heard)+[self.energy]) + list(actions)
            # sight,sight,sight,sound,sound,pattern,pattern,left,right,forward,chirp,chirp
            self.reponse_time_series.append(to_add)
            if len(self.reponse_time_series)>100:
                self.reponse_time_series = self.reponse_time_series[-100:]
            self.correlation = np.cov(np.array(self.reponse_time_series).T)

        if self.flapped>0:
            self.flapped-=1
            actions[2]=min(actions)-.01

        if self.chirped>0:
            self.chirped-=1
            for i in range(self.chirp_df):
                actions[3+i]=min(actions)-.01


        action = np.argmax(actions) #2 = flap

        if action == 0:
            self.velr = actions[action]*1.5
        elif action == 1:
            self.velr = -actions[action]*1.5
        elif action == 2:
            self.velr*=.85
            self.vel*=1.1
            self.energy-=1
            self.accel = 5+3*actions[action]
            self.flapped=12+random.randint(-10,10)
        else:
            self.energy-=3
            self.chirp(actions[3:4+self.chirp_df])
            self.chirped=20+random.randint(-10,15)


    def update_sound(self):
        a = math.radians(self.angle)
        self.facing_vector = [math.cos(a),math.sin(a)]
        if self.sound_timer>0:
            self.sound_timer-=1

            dist = ((self.x-self.sound_origin[0])**2 + (self.y-self.sound_origin[1])**2)**.5
            self.sound_direction = [-(self.x-self.sound_origin[0])/dist, -(self.y-self.sound_origin[1])/dist]

            a_dif=math.atan2(self.sound_direction[1],self.sound_direction[0])-math.atan2(self.facing_vector[1],self.facing_vector[0])
            self.sound_sensors = [max(0,a_dif),max(0,-a_dif)]
        else:
            self.sound_sensors = [0,0]

            self.sound_direction = [0,0]

    def update(self):
        self.update_sound()
        super().update()

    def draw(self):
        super().draw()

        pygame.gfxdraw.line(self.env.win, int(self.x), int(self.y), int(self.x+self.sound_direction[0]*15), int(self.y+self.sound_direction[1]*15), BLACK)

        if self.sound_timer>0:
            pygame.gfxdraw.line(self.env.win, int(self.x), int(self.y), int(self.x+self.facing_vector[0]*15), int(self.y+self.facing_vector[1]*15), BLACK)
            pygame.gfxdraw.circle(self.env.win, int(self.sound_origin[0]), int(self.sound_origin[1]), 8, PURPLE)

