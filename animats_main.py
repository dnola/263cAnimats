__author__ = 'davidnola'
import multiprocessing
import glob
from environment import Environment
import pickle

def load_social_birds():
    social_list = []
    print("loading social birds...")
    for f in glob.glob('bird_pickles/social*.pkl'):
        print(f)
        social_list+=pickle.load(open(f,'rb'))
    return social_list


def run_day(proc_id,warm_start=False):
    generation_time=5000
    display_on=125000
    sim_length = -1

    if not warm_start:
        if proc_id <= 0:
            e = Environment(enable_display=False,generation_time=generation_time,display_on=display_on,sim_length=sim_length,sim_id=proc_id)
        else:
            e = Environment(enable_display=False,generation_time=generation_time,display_on=None,sim_length=sim_length,sim_id=proc_id)
    else:
        social_list = load_social_birds()
        e = Environment(enable_display=False,generation_time=generation_time,display_on=display_on,sim_length=-1,sim_id=proc_id,social_bird_pool=social_list)

def main():
    run_day(-1)
    # jobs = [multiprocessing.Process(target=run_day,args=(i,)) for i in range(4)]
    # list(map(lambda x:x.start(), jobs))
    # list(map(lambda x:x.join(), jobs))
    # print("\n\n\n\n\n\n Running Final Sim:\n")
    # run_day(-1,True)


main()


