import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

plt.style.use('ggplot')

if __name__ == "__main__":

    results_dir = os.path.join(os.getcwd(),"..","results")
    costJ_dir = os.path.join(results_dir,"costJ")

    j = pickle.load(open(os.path.join(costJ_dir,"costJ.pickle"),"rb"))

    for image in tqdm(range(len(j))):

        plt.figure(figsize=(5,5))

        plt.plot(j[0:image])

        plt.xlabel("Time")
        plt.ylabel("CostJ")
        plt.title("Time - Cost J")

        # plt.legend()
        plt.tight_layout()

        file_out = os.path.join(costJ_dir,f"costJ_time_{image}.png")
        plt.savefig(file_out)
        plt.close()
