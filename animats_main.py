__author__ = 'davidnola'
import multiprocessing

from environment import Environment

def run_day(proc_id):
    generation_time=5000
    display_on=200000
    sim_length = 150000

    if proc_id == 0:
        e = Environment(enable_display=False,generation_time=generation_time,display_on=display_on)
    else:
        e = Environment(enable_display=False,generation_time=generation_time,display_on=None)

def main():
    jobs = [multiprocessing.Process(target=run_day,args=(i,)) for i in range(1)]
    list(map(lambda x:x.start(), jobs))


main()


