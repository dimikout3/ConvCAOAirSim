import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

plt.style.use('ggplot')

if __name__ == "__main__":

    results_dir = os.path.join(os.getcwd(),"..","results")
    similarity_dir = os.path.join(results_dir,"similarity_objects")

    file_avg = os.path.join(similarity_dir,"SimilarityList.pickle")
    avgSimilarity = pickle.load(open(file_avg,"rb"))

    avgSimilarity = np.array(avgSimilarity)

    for image in tqdm(range(len(avgSimilarity))):

        plt.figure(figsize=(5,5))

        x = [i for i in range((image+1))]
        yAvg = avgSimilarity[0:(image+1)]

        plt.plot(x,yAvg,label="Average")

        plt.xlabel("Time")
        plt.ylabel("Similarity KPI")
        plt.title("Mutual Information")

        plt.legend()
        plt.tight_layout()

        file_out = os.path.join(similarity_dir,f"similarity_time_{image}.png")
        plt.savefig(file_out)
        plt.close()
