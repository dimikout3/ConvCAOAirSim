# Semi-Exhaustive Search #

Thi is an algorithm that has little practical value, however, its achieved performance can provide us with valuable insights, when compared with the any other proposed navigation algorithm. This algorithm is a centralized, semi-exhaustive methodology that works as follows: At each timestamp, first, it generates a subset (semi-exhaustive) of candidate UAVs’ configurations (centralized) out of all possible ones. Then, all these candidates are evaluated on the AirSim platform, i.e. the UAVs have to actually reach that candidate monitoring positions, and, for each one of them, is calculated also the objective function (following the computation scheme from subsection II-C). Finally, the next configuration for the swarm is the candidate maximizes the objective function value. This procedure is repeated for every timestamp of the experiment.

# Specific App Instructions #

to be provided
