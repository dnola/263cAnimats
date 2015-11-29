import random

import pandas
import pygame
import pygame.gfxdraw

from constants import *
from social_bird import SocialBird
from predator_bird import PredatorBird
from generic_bird import Bird

class Environment:
    def __init__(self,enable_display=False,generation_time=5000,display_on=100000,social_bird_count=16,generic_bird_count=0,predator_bird_count=0):
        self.social_birds = []
        self.generic_birds = []
        self.predator_birds = []
        self.iterations = 0
        self.enable_display=enable_display
        self.generation_time=generation_time
        self.display_on = display_on
        self.trees = []
        self.env_size = 800
        self.grid_size = 10
        self.social_bird_count = social_bird_count
        self.generic_bird_count = generic_bird_count
        self.predator_bird_count = predator_bird_count
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
        for bird in self.social_birds+self.generic_birds+self.predator_birds:
            bird.draw()

    def update_birds(self):
        for bird in self.social_birds+self.generic_birds+self.predator_birds:
            bird.update()

    def seed_env(self):
        for i in range(self.social_bird_count):
            b = SocialBird(random.random()*self.env_size, random.random()*self.env_size, self)
            self.social_birds.append(b)

        for i in range(self.generic_bird_count):
            b = Bird(random.random()*self.env_size, random.random()*self.env_size, self)
            self.generic_birds.append(b)

        for i in range(self.predator_bird_count):
            b = PredatorBird(random.random()*self.env_size, random.random()*self.env_size, self)
            self.predator_birds.append(b)

        for i in range(10):
            t = Tree(int(random.random()*self.grid_width),int(random.random()*self.grid_width),self)
            self.trees.append(t)


    def breed_bird(self,bird_list):
        count = len(bird_list)
        sqrt = int(len(bird_list)**.5)
        b_1 = int(abs(random.gauss(0,sqrt)))
        b_2 = int(abs(random.gauss(0,sqrt)))
        while b_1 > count or b_2 > count:
            b_1 = int(abs(random.gauss(0,sqrt)))
            b_2 = int(abs(random.gauss(0,sqrt)))
        while b_1==b_2 or b_2 > count:
            b_2 = int(abs(random.gauss(0,sqrt)))

        return bird_list[b_1].breed(bird_list[b_2])

    def collect_statistics(self):
        corr = self.social_birds[0].correlation
        headers = ["sight","sight","sight","sound","sound" ,"pattern","pattern","energy","left","right","forward","chirp","chirp"]
        df = pandas.DataFrame(data=corr,index=headers,columns=headers)
        df = df.fillna(0)
        for bird in self.social_birds[1:]:
            df+=pandas.DataFrame(data=bird.correlation,index=headers,columns=headers).fillna(0)
        df = df/float(len(self.social_birds))
        print(df)

    def sort_birds(self):
        self.social_birds = list(reversed(sorted(self.social_birds)))
        self.generic_birds = list(reversed(sorted(self.generic_birds)))
        self.predator_birds = list(reversed(sorted(self.predator_birds)))
        try:
            print("Social:",self.social_birds[0].fitness,self.social_birds[-1].fitness)
            print("Predator:",self.predator_birds[0].fitness,self.predator_birds[-1].fitness)
            print("Generic:",self.generic_birds[0].fitness,self.generic_birds[-1].fitness)
        except:
            pass

    def reset_birds(self):
        fit = 0
        for bird in self.social_birds:
            fit+=bird.fitness
            bird.fitness=0 #max(0,bird.fitness)
            bird.energy=50
        print("Average Fitness:",fit/len(self.social_birds))

    def global_panmixia(self):
        self.sort_birds()
        self.collect_statistics()

        for i in range(int(self.social_bird_count**.5)):
            self.social_birds[-i+1]=self.breed_bird(self.social_birds)
        self.social_birds[-1]=SocialBird(random.random()*self.env_size, random.random()*self.env_size, self)

        self.reset_birds()

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
            #############

            if self.iterations%self.generation_time==0:
                self.global_panmixia()

            ############
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


