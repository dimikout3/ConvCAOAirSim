import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

plt.style.use('ggplot')

if __name__ == "__main__":

    results_dir = os.path.join(os.getcwd(),"..","results")
    information_dir = os.path.join(results_dir,"information")

    dronesList = os.listdir(information_dir)

    scoreDrones = []
    for drone in dronesList:

        if (drone == "scoreAggregated.pickle") or ("information" in drone):
            continue

        file_scoreDrone = os.path.join(information_dir,drone)
        scoreDrones.append(pickle.load(open(file_scoreDrone,"rb")))

    aggregatedExists = "scoreAggregated.pickle" in dronesList

    if aggregatedExists:
        file_aggregatedScore = os.path.join(information_dir,"scoreAggregated.pickle")
        aggregatedScore = pickle.load(open(file_aggregatedScore,"rb"))
    else:
        # combine each individual drone score for generatin one aggregated
        aggregatedScore = []

        for pos in range(len(scoreDrones[0])):
            sum = 0.0
            for score in scoreDrones:
                sum += score[pos]

            aggregatedScore.append(sum)

    for image in tqdm(range(len(aggregatedScore))):

        plt.figure(figsize=(5,5))

        x = [i for i in range((image+1))]
        yAggregated = aggregatedScore[0:(image+1)]

        plt.plot(x,yAggregated,label="Aggregated")

        # for scoreIdx,score in enumerate(scoreDrones):
        #     plt.plot(x,score[0:(image+1)], label=f"Drone{scoreIdx}")

        plt.xlabel("Time")
        plt.ylabel("Weighted Information")
        plt.title("Information")

        plt.legend()
        plt.tight_layout()

        file_out = os.path.join(information_dir,f"information_time_{image}.png")
        plt.savefig(file_out)
        plt.close()
