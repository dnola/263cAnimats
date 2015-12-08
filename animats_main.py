__author__ = 'davidnola'
import multiprocessing
import glob
from environment import Environment
import pickle


def load_social_birds():  # Loads social birds from the pretraining experiments
    social_list = []
    print("loading social birds...")
    for f in glob.glob('bird_pickles/social*.pkl'):  #
        print(f)
        social_list += pickle.load(open(f, 'rb'))
    return social_list


def run_day(proc_id, warm_start=False):
    """
    Runs a sin gle step of the simulation, either one of the pretraining
    steps when warm start is false, or a final step when warm start is true.
    Proc ID determines where the instrumentation data and bird data will be
    stored on disk
    """
    generation_time = 2000  # number of iterations within each generation
    display_on = 2000001  # When to turn on display. In this case,
    # display time > length of the simulation, so it never happens
    sim_length = 200000  # Number of iterations in the simulation

    if not warm_start:
        if proc_id <= 0:
            e = Environment(enable_display=False,
                            generation_time=generation_time,
                            display_on=display_on, sim_length=sim_length,
                            sim_id=proc_id)
        else:
            e = Environment(enable_display=False,
                            generation_time=generation_time, display_on=None,
                            sim_length=sim_length, sim_id=proc_id)
    else:
        generation_time = 10000
        display_on = 100000  #  Now display will happen before simulation
        # ends so we can observe final results
        sim_length = 250000
        social_list = load_social_birds()
        e = Environment(enable_display=False, generation_time=generation_time,
                        display_on=display_on, sim_length=-1, sim_id=proc_id,
                        social_bird_pool=social_list)


def main():
    jobs = [multiprocessing.Process(target=run_day, args=(i,)) for i in
            range(8)]  # Run 8 pretraining steps simultaneously
    list(map(lambda x: x.start(), jobs))
    list(map(lambda x: x.join(), jobs))  # Wait for pretraining to finish
    print("\n\n\n\n\n\n Running Final Sim:\n")
    run_day(-1, True)  # Run the final simulation


main() # Runs driver script
