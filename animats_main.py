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
import pandas
import multiprocessing


from constants import *



from environment import Environment

def run_day(proc_id):
    generation_time=5000
    display_on=120000

    if proc_id == 0:
        e = Environment(enable_display=False,generation_time=generation_time,display_on=display_on)
    else:
        e = Environment(enable_display=False,generation_time=generation_time,display_on=None)

def main():
    jobs = [multiprocessing.Process(target=run_day,args=(i,)) for i in range(2)]
    list(map(lambda x:x.start(), jobs))


main()


