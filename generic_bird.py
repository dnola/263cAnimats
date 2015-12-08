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
    def __init__(self, x, y, env, angle=0, weights=-1):

        self.my_id = int(random.randint(0, 999999))
        self.env = env

        n = pybrain.tools.shortcuts.buildNetwork(4, 5, 3,
                                                 hiddenclass=psm.LinearLayer,
                                                 outclass=psm.SigmoidLayer,
                                                 outputbias=True,
                                                 recurrent=True)
        if weights != -1:
            n._setParameters(weights)
        self.network = n

        self.fitness = 0
        self.energy = 50

        self.x = x
        self.y = y
        self.vel = 0
        self.accel = 0
        self.velr = 0
        self.angle = angle

        self.sight_rays = [[], [], []]
        self.sight_sensors = [0, 0, 0]
        self.seen_predator = [0]
        self.num_sight_pts = 12

        self.flapped = 0

    def breed(self, other):
        """
        Breed is called using two birds of the same class. It performs a
        single crossover recombination with a 10% chance of mutation - which
        is intentionally a high rate to encourage genetic diversity
        """
        e = self.env
        w1 = self.network.params # get the two sets of weights
        w2 = self.network.params

        crossover = int(random.random() * len(w1)) # pick a cross over point
        combined = list(w1[:crossover]) + list(w2[crossover:]) # splice the
        # weights
        while random.random() < .1:
            combined[int(random.random() * len(w1))] = random.gauss(0, np.max(
                np.fabs(combined))) # randomly permute some weights according
            #  to a standard deviation that is based on the maximum of the
            # network. This causes mutations to be roughly the same order as
            # the weights

        b = self.__class__(random.random() * e.env_size,
                           random.random() * e.env_size, e, weights=combined)
        #actually creates the new object, be it a Social, Generic,
        # or Predator bird
        return b

    def move(self):
        a = math.radians(self.angle)
        self.vel *= .8
        self.velr *= .95
        self.vel += self.accel * .1
        self.vel = min(15, self.vel)
        self.vel = max(0.0001, self.vel)
        self.accel = self.accel - .5
        self.x = self.wrap(self.x + self.vel * math.cos(a))
        self.y = self.wrap(self.y + self.vel * math.sin(a))
        self.angle += self.velr

    def run_network(self):
        actions = self.network.activate(self.sight_sensors + self.seen_predator)
        action = np.argmax(actions)

        if action == 0:
            self.velr = actions[action]
        elif action == 1:
            self.velr = -actions[action]
        else:
            self.velr *= .85
            self.energy -= .1
            self.accel = 3
            self.flapped = 20

    def get_nearby_birds(self):
        """
        Returns a list of birds "within range" of the bird
        """
        nearby = []
        for bird in self.env.social_birds:
            if bird.my_id == self.my_id: # don't include self, otherwise
                # birds would hear their own chirps and get confused
                continue

            dist = ((self.x - bird.x) ** 2 + (self.y - bird.y) ** 2) ** .5
            # euclidean distance
            if dist < 225:
                nearby.append(bird)

        return nearby

    def eat(self, tree_id):
        """
        Called when a bird attempts to eat food from a tree. If he is
        capable, he eats, and all nearby birds get a boost to fitness
        """
        if self.energy < 50: # If the bird is hungry...
            self.vel *= .5 # Swoop down to grab it (slow down)
            self.env.trees[tree_id].bite() # bite it
            self.energy = 100 # bird is now full
            for bird in self.get_nearby_birds(): # nearby birds gain .5 fitness
                self.fitness += .5
            self.fitness += 1 # This bird gains 1 fitness

    def see_food(self, food_coords):
        """
        uses scipy's cdist and numpy's argmin to rapidly calculate distances
        between sight rays and food
        """
        for idx, ray in enumerate(self.sight_rays):
            distances = cdist(ray, food_coords) # pairwise distance
            min = np.argmin(distances, axis=1) # gets closest tree to each
            #  ray point
            for i in range(self.num_sight_pts): # go through each sight point
                #  in the ray
                lmin = min[i] #gets the closest tree to THIS ray point

                if distances[0, lmin] < 15: #if it is close to bird, eat it
                    self.eat(lmin)

                if distances[i, lmin] < 12: # if it is close to sight ray,
                    # set the sight neuron
                    self.sight_sensors[idx] = self.num_sight_pts - i
                    break # sight has been obfuscated, stop tracing this ray

    def init_sight(self):
        a = math.radians(self.angle)
        offset = math.radians(12)

        self.sight_rays[0] = np.array([(
                                       self.wrap(self.x + 15 * i * math.cos(a)),
                                       self.wrap(self.y + 15 * i * math.sin(a)))
                                       for i in range(self.num_sight_pts)])
        self.sight_rays[1] = np.array([(self.wrap(
            self.x + 15 * i * math.cos(a + offset)), self.wrap(
            self.y + 15 * i * math.sin(a + offset))) for i in
                                       range(self.num_sight_pts)])
        self.sight_rays[2] = np.array([(self.wrap(
            self.x + 15 * i * math.cos(a - offset)), self.wrap(
            self.y + 15 * i * math.sin(a - offset))) for i in
                                       range(self.num_sight_pts)])

        self.sight_sensors = [0, 0, 0]

    def see_predator(self):
        pred_coords = [(p.x, p.y) for p in self.env.predator_birds]
        if self.seen_predator[0] > 0:
            self.seen_predator[0] -= .01
        for idx, ray in enumerate(self.sight_rays):
            distances = cdist(ray, pred_coords)

            for i in range(self.num_sight_pts):
                lmin = np.argmin(distances[i, :])
                if distances[i, lmin] < 15:
                    if self.sight_sensors[idx] == 0 or self.num_sight_pts - \
                            self.sight_sensors[idx] >= i:
                        self.sight_sensors[idx] = -(self.num_sight_pts - i)
                        self.seen_predator = [1]
                    break

    def update_sight_and_contact(self):
        self.init_sight()
        self.see_food([[t.x, t.y] for t in self.env.trees])
        self.see_predator()

    def wrap(self, x):
        return x % self.env.env_size

    def update(self):
        self.energy -= .2

        if self.energy < 0:
            self.energy *= .95
            self.fitness += self.energy * .0005
        self.move()

        self.update_sight_and_contact()

        self.run_network()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def draw(self):
        a = math.radians(self.angle)

        pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 5, (
        30, min(255 - int(255 * self.energy / 100.0), 255),
        min(255 - int(254 * self.energy / 100.0), 255)))
        if self.seen_predator[0] > 0:
            pygame.gfxdraw.aacircle(self.env.win, int(self.x), int(self.y), 2,
                                    DARK_RED)

        pygame.draw.aaline(self.env.win, BLACK, [int(self.x), int(self.y)],
                           [int(self.x + 7 * math.cos(a)),
                            int(self.y + 7 * math.sin(a))])

        for j, ray in enumerate(self.sight_rays):
            for i, (x, y) in enumerate(ray):
                if self.num_sight_pts - abs(self.sight_sensors[j]) < i:
                    break
                if self.energy > 0:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1, RED)
                else:
                    pygame.gfxdraw.circle(self.env.win, int(x), int(y), 1,
                                          DARK_RED)
