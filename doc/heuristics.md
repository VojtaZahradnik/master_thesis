# Heuristics algorithm

Heuristics algorithm is group of algorithms, that using logic of heuristic. I am using heuristics algorithm for optimization of loss function to get better 
prediction based on this function. We don't know the best loss function, because for every athlete is different, so for every athlete we need to optimize this function.

Every algorithm is implemented in folder [heuristics](../../research_task/src/heuristics)

## Random shooting

RS is heuristic algorithm based on random variables. It just generates random loss function and try training and testing process. After evaluation of the model we get metric, that 
tells us how good model really is. RS is really not that bad how sounds it.

I am using RS for basic generation of loss functions for future optimization by Steepest descent or Genetic optimization. This process seems to be really effective for this optimization purpose. Even through the random chance could be very tricky.

## Steepest descent

I implemented Steepest descent as algorithm, that is used for optimization after RS. We have already loss function and we need to minimize number of variables. 
This algorithm walks through loss function and tries every variable. If variable minimize evaluation metric, will be in loss function, but if not the algorithm drop this variable.

## Genetic Optimization

The main optimization segment is GO. Implementation we can find in [genetic_optimization.py](../../research_task/src/heuristics/genetic_optimization.py). We needed to define multiple selection, crossover, mutation and whole optimization process. 
Every of those term is described in documentation. 

On input of GO we have:

- Number of generations: Iterations, that GO makes.
- Population size: Number of loss functions in list called population. This population is inserted into iterations of GO.
- Mutation coefficient: How much we want mutation between two loss functions.
- Crossover coefficient: How much we want crossover between two loss functions.
- Training dataset
- Testing dataset
- Basic loss function
- Objective function

I provided test of GO and made 100 iterations of whole process of GO. After this I calculated similarity of loss functions after optimization and 
plotted heatmap. You can look to documentation for results of this research.