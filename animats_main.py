__author__ = 'davidnola'
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


pygame.init()
BLACK = (  0,   0,   1)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   1)
RED =   (255,   0,   1)
PURPLE =   (255,   0,   255)
DARK_RED =   (180,   0,   1)

class Environment:
    def __init__(self):
        self.social_birds = []
        self.iterations = 0
        self.enable_display=False
        self.trees = []
        self.env_size = 800
        self.grid_size = 10
        self.grid_width = self.env_size/self.grid_size
        self.win = pygame.display.set_mode((800,800))
        self.seed_env()
        self.draw_birds()
        self.run()

    def draw_env(self):
        self.win.fill(WHITE)

        for i in range(self.grid_size,self.env_size,self.grid_size):

            pygame.gfxdraw.hline(self.win, 0, self.env_size, i, (0,0,5,20))
            pygame.gfxdraw.vline(self.win, i, self.env_size, 0, (0,0,5,20))

        self.draw_trees()
        self.draw_birds()
        pygame.display.flip()

    def draw_trees(self):
        for tree in self.trees:
            tree.draw()

    def draw_birds(self):
        for bird in self.social_birds:
            bird.draw()

    def update_birds(self):
        for bird in self.social_birds:
            bird.update()

    def seed_env(self):
        for i in range(8):
            b = SocialBird(random.random()*500, random.random()*500, self)
            self.social_birds.append(b)
        for i in range(20):
            t = Tree(int(random.random()*self.grid_width),int(random.random()*self.grid_width),self)
            self.trees.append(t)

    def sort_birds(self):
        self.social_birds = list(reversed(sorted(self.social_birds)))
        print(self.social_birds[0].fitness,self.social_birds[-1].fitness)

    def reset_birds(self):
        fit = 0
        for bird in self.social_birds:
            fit+=bird.fitness
            bird.fitness=0
        print("Average Fitness:",fit/len(self.social_birds))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while(running):
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT:
                    running = False
            self.iterations+=1
            if self.iterations%100==0:
                print(self.iterations)

            if self.iterations%500==0:
                self.sort_birds()
                self.social_birds[-1]=self.social_birds[0].breed(self.social_birds[1])
                self.social_birds[-2]=self.social_birds[2].breed(self.social_birds[3])
                self.social_birds[-3]=self.social_birds[4].breed(self.social_birds[5])
                self.reset_birds()

            if self.iterations>2000:
                self.enable_display=True
            self.update_birds()

            if self.enable_display:
                self.draw_env()
                time.sleep(1)
        pygame.quit()


class Bird:
    def __init__(self,x,y,env,angle=0):
        self.x=x
        self.y=y
        self.vel = 0
        self.accel=0
        self.velr = 0
        self.angle=angle
        self.env=env
        self.fitness = 0

    def wrap(self,x):
        return x%self.env.env_size

    def update(self):
        pass

    def __lt__(self, other):
        return self.fitness < other.fitness

    def draw(self):
        a = math.radians(self.angle)

        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 3, BLACK)
        pygame.draw.aaline(self.env.win, BLACK, [int(self.x), int(self.y)], [int(self.x+7*math.cos(a)), int(self.y+7*math.sin(a))])

class Tree:
    def __init__(self,x,y,env,food=3,max_food=3):
        self.x=env.grid_size*int(x)+5
        self.y=env.grid_size*int(y)+5
        self.env=env
        self.food=food
        self.max_food=max_food

    def reseed(self):
        self.x=self.env.grid_size*int(int(random.random()*self.env.grid_width))+5
        self.y=self.env.grid_size*int(int(random.random()*self.env.grid_width))+5

    def bite(self):
        self.food-=1
        pygame.gfxdraw.filled_circle(self.env.win, self.x, self.y, 5, BLACK)
        if self.food<=0:
            self.food=self.max_food
            self.reseed()

    def draw(self):
        pygame.gfxdraw.filled_circle(self.env.win, self.x, self.y, 5, GREEN)


class SocialBird(Bird):
    def __init__(self,x,y,env,weights=-1,angle=0):
        super().__init__(x,y,env=env,angle=angle)
        n = pybrain.tools.shortcuts.buildNetwork(5,5,5,hiddenclass=psm.LSTMLayer, outclass=psm.SoftmaxLayer, outputbias=False, recurrent=True)
        if weights!=-1:
            n._setParameters(weights)
        self.network = n
        self.sight_rays = [[],[],[]]
        self.sight_sensors = [0,0,0]
        self.sound_origin = [x,y]
        self.sound_sensors = [0,0]
        self.flapped = 0
        self.energy = 100
        self.facing_vector = [0,0]

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

        b = SocialBird(random.random()*e.env_size,random.random()*e.env_size,e,weights=combined)
        return b

    def chirp(self,sn):
        for bird in self.env.social_birds:
            if bird!=self:
                bird.heard_sound(self.x,self.y,sn)

    def heard_sound(self,sx,sy,sn):
        self.sound_origin = [sx,sy]

    def run_network(self):
        actions = self.network.activate(self.sight_sensors+self.sound_sensors)

        if self.flapped>0:
            self.flapped-=1
            actions[2]=0

        action = np.argmax(actions) #2 = flap

        if action == 0:
            self.velr = actions[action]
        elif action == 1:
            self.velr = -actions[action]
        elif action == 2 or action==3:
            self.chirp([actions[2],actions[3]])
        else:
            self.velr*=.5
            self.energy-=1
            self.accel = 1

    def move(self):
        a = math.radians(self.angle)
        self.vel*=.95
        self.vel+=self.accel
        self.vel = min(5,self.vel)
        self.vel = max(.5,self.vel)
        self.accel=self.accel-.01
        self.x = self.wrap(self.x+ self.vel * math.cos(a))
        self.y = self.wrap(self.y+ self.vel * math.sin(a))
        self.angle += self.velr

    def eat(self,tree_id):
        if self.energy<50:
            self.env.trees[tree_id].bite()
            self.energy=100
        self.fitness+=1

        for bird in self.env.social_birds:
            self.fitness+=.1

    def update_sight_and_contact(self):
        a = math.radians(self.angle)
        offset = math.radians(10)
        num_sight_pts = 15
        self.sight_rays[0] = np.array([(self.wrap(self.x+8*i*math.cos(a)),self.wrap(self.y + 8*i*math.sin(a))) for i in range(num_sight_pts)])
        self.sight_rays[1] = np.array([(self.wrap(self.x+8*i*math.cos(a+offset)),self.wrap(self.y + 8*i*math.sin(a+offset))) for i in range(num_sight_pts)])
        self.sight_rays[2] = np.array([(self.wrap(self.x+8*i*math.cos(a-offset)),self.wrap(self.y + 8*i*math.sin(a-offset))) for i in range(num_sight_pts)])

        tree_coords = [[t.x,t.y] for t in self.env.trees]


        self.sight_sensors = [0,0,0]
        for idx,ray in enumerate(self.sight_rays):
            distances = cdist(ray,tree_coords)

            for i in range(num_sight_pts):
                lmin = np.argmin(distances[i,:])

                if distances[i,lmin] < 10:
                    if i==0:
                        self.eat(lmin)
                    self.sight_sensors[idx]=i
                    break

    def update_sound(self):
        a = math.radians(self.angle)

        dist = ((self.x-self.sound_origin[0])**2 + (self.y-self.sound_origin[1])**2)**.5
        self.sound_sensors = [-(self.x-self.sound_origin[0])/dist, -(self.y-self.sound_origin[1])/dist]
        self.facing_vector = [math.cos(a),math.sin(a)]

        ### TODO: Figure out how to get sound sensors to help direct bird to sound ###



    def update(self):

        self.energy-=.01

        if self.energy<0:
            self.fitness+=self.energy*.1
        self.move()
        self.update_sight_and_contact()
        self.update_sound()

        self.run_network()

    def draw(self):
        super().draw()

        pygame.gfxdraw.line(self.env.win, int(self.x), int(self.y), int(self.x+self.sound_sensors[0]*15), int(self.y+self.sound_sensors[1]*15), BLACK)
        pygame.gfxdraw.line(self.env.win, int(self.x), int(self.y), int(self.x+self.facing_vector[0]*15), int(self.y+self.facing_vector[1]*15), BLACK)


        pygame.gfxdraw.circle(self.env.win, int(self.sound_origin[0]), int(self.sound_origin[1]), 8, PURPLE)

        for ray in self.sight_rays:
            for (x,y) in ray:
                if self.energy>0:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, RED)
                else:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, DARK_RED)

def main():
    e = Environment()



main()