import random

import pandas
import pygame
import pygame.gfxdraw

from constants import *
from social_bird import SocialBird

class Environment:
    def __init__(self,enable_display=False,generation_time=5000,display_on=100000):
        self.social_birds = []
        self.iterations = 0
        self.enable_display=enable_display
        self.generation_time=generation_time
        self.display_on = display_on
        self.trees = []
        self.env_size = 800
        self.grid_size = 10
        self.grid_width = self.env_size/self.grid_size
        self.win=None
        if self.win==None and enable_display:
            pygame.init()
            self.win = pygame.display.set_mode((800,800))
        self.seed_env()
        self.draw_env()
        self.run()

    def draw_env(self):
        if self.enable_display:
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
        for i in range(16):
            b = SocialBird(random.random()*500, random.random()*500, self)
            self.social_birds.append(b)
        for i in range(10):
            t = Tree(int(random.random()*self.grid_width),int(random.random()*self.grid_width),self)
            self.trees.append(t)

    def sort_birds(self):
        self.social_birds = list(reversed(sorted(self.social_birds)))
        corr = self.social_birds[0].correlation
        headers = ["sight","sight","sight","sound","sound" ,"pattern","pattern","energy","left","right","forward","chirp","chirp"]
        df = pandas.DataFrame(data=corr,index=headers,columns=headers)
        df = df.fillna(0)
        for bird in self.social_birds[1:]:
            df+=pandas.DataFrame(data=bird.correlation,index=headers,columns=headers).fillna(0)
        df = df/float(len(self.social_birds))
        print(df)

        print(self.social_birds[0].fitness,self.social_birds[-1].fitness)

    def reset_birds(self):
        fit = 0
        for bird in self.social_birds:
            fit+=bird.fitness
            bird.fitness=0 #max(0,bird.fitness)
            bird.energy=50
        print("Average Fitness:",fit/len(self.social_birds))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while(running):
            if self.enable_display:
                for event in pygame.event.get(): # User did something
                    if event.type == pygame.QUIT:
                        running = False
            self.iterations+=1
            if self.iterations%100==0:
                print(self.iterations)

            if self.iterations%self.generation_time==0:
                self.sort_birds()
                try:
                    self.social_birds[-1]=self.social_birds[0].breed(self.social_birds[1])
                    self.social_birds[-2]=self.social_birds[2].breed(self.social_birds[3])
                    self.social_birds[-3]=self.social_birds[4].breed(self.social_birds[5])
                    self.social_birds[-4]=SocialBird()
                    self.social_birds[-5]=SocialBird()
                except:
                    pass
                self.reset_birds()

            if self.display_on!= None and self.iterations>self.display_on:

                if self.win==None:
                    pygame.init()
                    self.win = pygame.display.set_mode((800,800))
                self.enable_display=True
            self.update_birds()

            if self.enable_display:
                self.draw_env()
                # time.sleep(1)
        pygame.quit()



class Tree:
    def __init__(self,x,y,env,max_food=5):
        self.x=env.grid_size*int(x)+5
        self.y=env.grid_size*int(y)+5
        self.env=env
        self.food=max_food
        self.max_food=max_food

    def reseed(self):
        self.x=self.env.grid_size*int(int(random.random()*self.env.grid_width))+5
        self.y=self.env.grid_size*int(int(random.random()*self.env.grid_width))+5

    def bite(self):
        self.food-=1
        print("bite")
        if self.food<=0:
            self.food=self.max_food
            self.reseed()

    def draw(self):
        if self.env.enable_display:
            pygame.gfxdraw.filled_circle(self.env.win, self.x, self.y, 5, (  0, 55+int(200*self.food/self.max_food),   1))


