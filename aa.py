from SynapEvo.NNevo import Population
import SynapEvo.NN as NN
import matplotlib.pyplot as plt

input_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_set = [0, 1, 1, 1]
Pop = Population(input_size=2, output_size=1, layers_sizes=2, nlayers=3, np_nr=0.70, population_size=100,
                 parent_percentage=0.10, mutation_rate=0.2)

ourpop = Pop.get_populations()
aths = 0
athss = None
hsgen = []
for i in range(100):
    # running for about 500 generations
    hs = 0
    for j in range(len(ourpop)):
        score = 4.0
        for k in range(len(input_set)):
            res = ourpop[j].forward(input_set[k])
            score -= (res - output_set[k]) ** 2
        ourpop[j].score = score
        if score > hs:
            hs = score
        if score > aths:
            aths = score
            athss = ourpop[j]
    hsgen.append(hs)
    ourpop = Pop.evolve(ourpop)

for i in input_set:
    print(athss.forward(i))
print(aths)
plt.plot(hsgen)
plt.show()