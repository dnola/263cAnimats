import math
import random
import numpy as np
import pybrain.structure.modules as psm
import pybrain.tools.shortcuts
import pygame
import pygame.gfxdraw
from constants import *
from generic_bird import Bird


class SocialBird(Bird): # Inherit from generic bird
    def __init__(self, x, y, env, weights=-1, angle=0, chirp_df=3):
        super().__init__(x, y, env=env, angle=angle)

        # Constructs the initial neural net with randomized weights
        n = pybrain.tools.shortcuts.buildNetwork(7 + chirp_df, 15, 12,
                                                 3 + chirp_df,
                                                 hiddenclass=psm.LinearLayer,
                                                 outclass=psm.SigmoidLayer,
                                                 outputbias=True,
                                                 recurrent=True)
        # If given a set of weights, load them into the network
        if weights != -1:
            n._setParameters(weights)
        self.network = n

        # To keep track of statistics
        self.reponse_time_series = []
        self.correlation = []

        self.sound_origin = [x, y] # Stores origin point of last sound hear
        self.sound_sensors = [0 , 0] # sound direction sensors
        self.sound_direction = [0, 0] # absolute direction of sound, used to
        # calculate sound_sensors
        self.chirp_df = chirp_df

        self.facing_vector = [0, 0] # stores current direction bird is
        # facing, used to calculate sound_sensors
        self.sound_timer = 0 # variable to determine when bird forgets sound
        self.pattern_heard = [0 for i in range(chirp_df)] # chirp sensors

        self.chirped = 0 # used to prevent bird from chirping too often

    def chirp(self, sn):
        """
        Calls heard_sound() on all nearby birds, passing current
        location and chirp pattern
        """
        for bird in self.get_nearby_birds():
            bird.heard_sound(self.x, self.y, sn)

    def heard_sound(self, sx, sy, sn):
        """
        Updates sensors based on chirp pattern and origin point of chirp,
        sets a timer for 300 iterations before forgettign sound origin
        """
        self.sound_origin = [sx, sy]
        self.pattern_heard = sn
        self.sound_timer = 300

    def run_network(self):
        """
        Runs the neural network, gets the max output, collects a
        sensorimotor covariance matrix (sometimes) and performs the action
        """
        actions = self.network.activate(
            list(self.sight_sensors) + self.seen_predator + list(
                self.sound_sensors) + list(self.pattern_heard) + [
                self.energy - 30]) # activates RNN. Note the energy offset -
        # this is so the energy input goes negative before the bird actually
        # starves. The negative polarity is used sometimes by the birds to
        # induce 'desperate' actions, like running around in a last ditch
        # effort - so it helps for this to happen before actual starvation

        if random.random() > .9 or len(self.reponse_time_series) < 3: # We
            # only collect covariance matrices 10% of the time because this
            # is expensive both in terms of computation and in terms of
            # memory, plus adjacent time steps tend to be pretty much
            # identical anyways
            to_add = list(list(self.sight_sensors) + self.seen_predator + list(
                self.sound_sensors) + list(self.pattern_heard) + [
                              self.energy]) + list(actions)

            self.reponse_time_series.append(to_add)
            if len(self.reponse_time_series) > 100: # limit size of each
                # birds covariance storage to prevent memory crash
                self.reponse_time_series = self.reponse_time_series[-100:]
            self.correlation = np.cov(np.array(self.reponse_time_series).T)

        if self.flapped > 0: # prevent bird from flapping wings too often -
            # makes bird movement look more realistic
            self.flapped -= 1
            actions[2] = min(actions) - .01

        if self.chirped > 0: # prevent bird from chirping too often - stops
            # birds from being overwhelmed by too many chirps to respond to
            self.chirped -= 1
            for i in range(self.chirp_df):
                actions[3 + i] = min(actions) - .01

        action = np.argmax(actions)  # 2 = flap

        if action == 0: # turn left
            self.velr = actions[action] * 1.5
        elif action == 1: # turn right
            self.velr = -actions[action] * 1.5
        elif action == 2: # flap wings
            self.velr *= .85 # slows rotation
            self.vel *= 1.1 # immediate boost
            self.energy -= 1 # energy cost
            self.accel = 5 + 3 * actions[action] # add a quick acceleration
            # as bird flaps - disapates quickly
            self.flapped = 12 + random.randint(-10, 10) # prevent bird from
            # flapping immediately again
        else: # if none of the above are maximum, then chirp is selected
            self.energy -= 10 # chirps are very costly to prevent overuse -
            # birds should learn to only chirp with purpose
            self.chirp(actions[3:4 + self.chirp_df]) # copy into nearby birds
            self.chirped = 60 + random.randint(-20, 50) # prevent too much
            # chirping

    def update_sound(self):
        """
        Used to update sound_sensor continuously depending on how far bird
        would need to turn to face sound_origin
        """
        a = math.radians(self.angle)
        self.facing_vector = [math.cos(a), math.sin(a)] # get unit facing vector
        if self.sound_timer > 0: # only perform calculations if bird has a
            # sound in memory
            self.sound_timer -= 1 # forget sound over time


            # now get a unit vector in the direction of the sound
            dist = ((self.x - self.sound_origin[0]) ** 2 + (
            self.y - self.sound_origin[1]) ** 2) ** .5
            self.sound_direction = [-(self.x - self.sound_origin[0]) / dist,
                                    -(self.y - self.sound_origin[1]) / dist]

            # use atan2 to get the difference between facing vector and
            # direction vector into radians
            a_dif = math.atan2(self.sound_direction[1],
                               self.sound_direction[0]) - math.atan2(
                self.facing_vector[1], self.facing_vector[0])

            self.sound_sensors = [max(0, a_dif), max(0, -a_dif)] # positive
            # means left, negative means right
        else:
            self.sound_sensors = [0, 0]

            self.sound_direction = [0, 0]

    def update(self):
        self.update_sound()
        super().update()

    def draw(self):
        super().draw() # draws the bird

        if self.sound_timer > 0:
            pygame.gfxdraw.line(self.env.win, int(self.x), int(self.y),
                            int(self.x + self.sound_direction[0] * 15),
                            int(self.y + self.sound_direction[1] * 15),
                            BLACK) # draws the sound direction vector


            pygame.gfxdraw.line(self.env.win, int(self.x), int(self.y),
                                int(self.x + self.facing_vector[0] * 15),
                                int(self.y + self.facing_vector[1] * 15),
                                BLACK) #draws the facing direction vector
            pygame.gfxdraw.circle(self.env.win, int(self.sound_origin[0]),
                                  int(self.sound_origin[1]), 8, PURPLE) #
            # draws the origin point of the sound the bird is currently
            # paying attention to
