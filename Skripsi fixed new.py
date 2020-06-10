import random
import networkx as nx

# for basic visualizations
import matplotlib.pyplot as plt

start_node = 1
pop_size = 6
generation = 20
probabilitas_OX = 0.7
Mutation_probability = 0.9

filename = 'input.txt'
edgelist = open(filename, 'r')

positionfile = 'input_position.txt'
positionlist = open(positionfile, 'r')

yourResult = [line.strip().split(' ') for line in edgelist.readlines()]
yourResult

yourPosition = [line.strip().split(' ') for line in positionlist.readlines()]
yourPosition

edgelist = [[int(x) for x in lst] for lst in yourResult]
positionlist = [[int(x) for x in lst] for lst in yourPosition]

def make_graph(edgelist):
    graph = {}
    for e1, e2, weight in edgelist:
        graph.setdefault(e1, {}).setdefault(e2, weight)
        graph.setdefault(e2, {}).setdefault(e1, weight)
    return graph

graph = make_graph(edgelist)
graph

def make_individual(start, graph):
#Initialize a multicast tree
    start = start
    items = random.choice(list(graph[start].items()))
    end = items[0]
    weight = items[1]
    mst = []
    mst.append([start, end, weight])
    circuit_start = []
    circuit_end = []
    circuit_end.append(end)

    closed = []
    closed.append([start, end])
    closed.append([end, start])
    circuit_start.append(start)
    i = 1
    #Generate random path which from the tree T to the destination node d as RunPath(u, d)
    while (i < len(graph) - 1):
        start = items[0]
        items = random.choice(list(graph[start].items()))
        end = items[0]
        
        weight = items[1]
        if [start, end] in closed:
            continue
        if end in circuit_start:
            continue 
        if end in circuit_end:
            continue
            
        mst.append([start, end, weight])
        circuit_start.append(end)
        circuit_end.append(end)
        closed.append([start, end])
        closed.append([end, start])
        i = i+1

    assert len(mst) == len(graph) - 1
    return mst


def counting_weight(mst):
    bobot = 0
    for i in range(len(mst)):
        bobot = bobot+mst[i][2]
    return bobot

fixed_position = {}
for nodes, x, y in positionlist:
    position = (x, y)
    fixed_position.setdefault(nodes, position)
fixed_position_nodes = fixed_position.keys()

def make_network_viz(graph):
    g = nx.Graph()
    for edge in graph:
        g.add_edge(edge[0],edge[1], weight = edge[2])

    import warnings
    warnings.filterwarnings('ignore')

    plt.rcParams['figure.figsize'] = (11, 6)
    plt.style.use('fivethirtyeight')

    pos = nx.spring_layout(g, pos = fixed_position, fixed = fixed_position_nodes, weight = 'weight')

    # drawing nodes
    nx.draw_networkx(g, pos, arrows = True, node_color='green', node_size=350)
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in g.edges(data=True)])
    nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels, font_color='green', font_size=10)
    #plt.title('Undirected Graphs', fontsize = 20)
    plt.axis('on')
    plt.show()

graph_viz = make_network_viz(edgelist)

mst = make_individual(1, graph)

individual = make_network_viz(mst)

make_graph(mst)

def create_population(popSize, startNode):
    popList = []
    for i in range(popSize):
        popList.append(make_individual(startNode, graph))
    return popList

def count_weight(popList):
    total_weight= []

    for i in popList:
        total_weight.append(counting_weight(i))

    return total_weight

def count_fitness(popList):
    objective= []

    for i in popList:
        objective.append(counting_weight(i))
    
    #total_fitness = float(sum(objective))
    rel_fitness = [1/(f+1) for f in objective]  # <- baru
#     rel_fitness = [f/total_fitness for f in objective] # <- lama
    return rel_fitness

def roulette_selection(popList):
    fitnesses = count_fitness(popList)
    total_fitness = float(sum(fitnesses))
    rel_fitness = [f/total_fitness for f in fitnesses]
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    new_population = []
    for n in range(pop_size):
        r = random.random()
        for (i, individual) in enumerate(popList):
            if r <= probs[i]:
                new_population.append(individual)
                break
    for i in new_population:
        print(i, 'bobot: ', counting_weight(i), '\n')
    return new_population

def PartialyMatched_Crossover(parent1, parent2):
    ind1 = parent1
    ind2 = parent2
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[i] = i
        p2[i] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1] = temp2, temp1
        ind2[i], ind2[p2] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

def Ordered_Crossover(parent1, parent2):
    ind1 = parent1
    ind2 = parent2
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[i] = False
            holes2[i] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[(i + b + 1) % size]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[(i + b + 1) % size]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

#untuk mencari nilai fitness terbaik dari suatu populasi

def elitis_selection(popList):
    fitnesses = count_fitness(popList)
    pop = []
    new_population = []
    for i in range(len(fitnesses)):
        pop.append([fitnesses[i], popList[i]])
        
    ranked_population = sorted(pop, reverse = True)
    
    for i in range (len(ranked_population)):
        new_population.append(ranked_population[i][1])

    return new_population

def crossover(probabilitas_OX, selected_pop):
    popRand = []
    acak = random.random()
    for i in range(len(selected_pop)):
        acak = random.random()
        popRand.append([acak, selected_pop[i]])
    
    sisa = []
    matingPool = []
    for i in range (len(popRand)):
        if popRand[i][0] < probabilitas_OX:
            matingPool.append(popRand[i][1])
        else:
            sisa.append(popRand[i][1])
            
    sum_rand = random.randint(len(mst)*30/100, len(mst)*90/100) #tidak ada pengaruh jika diubah
    point = []
    for i in range(len(mst)):
        point.append(i)

    x=0
    while x < len(matingPool):
        point = random.sample(point, sum_rand)
        temp1 = []
        temp2 = []
        circuit_start1 = []
        circuit_start2 = []
        swap = []
        closed1 = []
        closed2 = []

        circuit_start1.append(start_node)
        temp1.append(start_node)
        items1 = random.choice(list(graph[start_node].items()))
        circuit_start2.append(start_node)
        temp2.append(start_node)
        items2 = random.choice(list(graph[start_node].items()))

        i=1
        for i in range(len(graph)-1):
            if i in point:
                swap.append(matingPool[x][i])
                swap.append(matingPool[x+1][i])
                choose = random.choice(swap)
                if choose[1] in temp1:
                    j = 1
                    while j < len(matingPool[x][i]):
                        start1 = items1[0]
                        items1 = random.choice(list(graph[start1].items()))
                        end1 = items1[0]
                        weight1 = items1[1]
                        if [start1, end1] in closed1:
                            continue
                        if [end1, start1] in closed1:
                            continue
                        if start1 not in temp1:
                            continue
                        if end1 in temp1:
                            continue
                        matingPool[x][i] = [start1, end1, weight1]
                        j += 1
                    temp1.append(end1)
                    circuit_start1.append(start1)
                    closed1.append([start1, end1])
                    closed1.append([end1, start1])
                else:        
                    matingPool[x][i] = choose
                    closed1.append([matingPool[x][i][0], matingPool[x][i][1]])
                    closed1.append([matingPool[x][i][1], matingPool[x][i][0]])
                    start1 = matingPool[x][i][1]
                    temp1.append(matingPool[x][i][1])
                    circuit_start1.append(matingPool[x][i][0])
                swap.remove(choose)
                if swap[0][1] in temp2:
                    j = 1
                    matingPool[x+1][i] = swap[0]
                    while j < len(matingPool[x+1][i]):
                        start2 = items2[0]
                        items2 = random.choice(list(graph[start2].items()))
                        end2 = items2[0]
                        weight2 = items2[1]
                        if [start2, end2] in closed2:
                            continue
                        if [end2, start2] in closed2:
                            continue
                        if start2 not in temp2:
                            continue
                        if end2 in temp2:
                            continue
                        matingPool[x+1][i] = [start2, end2, weight2]
                        j += 1
                    circuit_start2.append(start2)
                    temp2.append(end2)
                    closed2.append([start2, end2])
                    closed2.append([end2, start2])
                else:
                    matingPool[x+1][i] = swap[0]
                    closed2.append([matingPool[x+1][i][0], matingPool[x+1][i][1]])
                    closed2.append([matingPool[x+1][i][1], matingPool[x+1][i][0]])
                    start2 = matingPool[x+1][i][1]
                    temp2.append(matingPool[x+1][i][1])
                    circuit_start2.append(matingPool[x+1][i][0])
                swap.clear()
            else:
                j = 1
                while j < len(graph):
                    start1 = items1[0]
                    items1 = random.choice(list(graph[start1].items()))
                    end1 = items1[0]
                    weight1 = items1[1]
                    if [start1, end1] in closed1:
                        continue
                    if [end1, start1] in closed1:
                        continue
                    if start1 not in temp1:
                        continue
                    if end1 in temp1:
                        continue
                    matingPool[x][i] = [start1, end1, weight1]
                    j += 1
                temp1.append(matingPool[x][i][1])
                circuit_start1.append(start1)
                closed1.append([start1, end1])
                closed1.append([end1, start1])

                j = 1
                while j < len(graph):
                    start2 = items2[0]
                    items2 = random.choice(list(graph[start2].items()))
                    end2 = items2[0]
                    weight2 = items2[1]
                    if [start2, end2] in closed2:
                        continue
                    if [end2, start2] in closed2:
                        continue
                    if start2 not in temp2:
                        continue
                    if end2 in temp2:
                        continue
                    matingPool[x+1][i] = [start2, end2, weight2]
                    j += 1
                temp2.append(matingPool[x+1][i][1])
                circuit_start2.append(start2)
                closed2.append([start2, end2])
                closed2.append([end2, start2])
        x += 2
        if x == len(matingPool) - 1:
            break
    return matingPool, sisa

def mutation1(Mutation_probability, popList):
    for i in popList:
        acak = random.random()
        if acak < Mutation_probability:
            temp = []
            closed = []
            end = []
            end.append(start_node)
            for j in i:
                temp.append(j)
                closed.append([j[0], j[1]])
                closed.append([j[1], j[0]])
                end.append(j[1])

            rand = random.randint(1, len(i)-1)
            end.remove(i[rand][1])

            j = 0
            while j < len(i):
                if j == rand:
                    nodes = random.choice(edgelist)
                    while nodes[1] in end:
                        nodes = random.choice(edgelist)
                        if [nodes[0], nodes[1]] in closed:
                            continue
                        if [nodes[1], nodes[0]] in closed:
                            continue
                        if nodes[1] in end:
                            continue
                        i = nodes
                j+=1
        else:
            continue
    return popList

def mutation(Mutation_probability, selected_pop):
    popRand = []
    acak = random.random()
    for i in range(len(selected_pop)):
        acak = random.random()
        popRand.append([acak, selected_pop[i]])

    mutationPool = []
    for i in range (len(popRand)):
        if popRand[i][0] < Mutation_probability:
            mutationPool.append(popRand[i][1])
        else:
            continue
    x = 0
    while x < len(mutationPool)-1:
        temp = []
        closed = []
        end = []
        end.append(start_node)
        for i in mutationPool[x]:
            temp.append(i)
            closed.append([i[0], i[1]])
            closed.append([i[1], i[0]])
            end.append(i[1])

        rand = random.randint(1, len(mutationPool[x])-1)
        end.remove(mutationPool[x][rand][1])

        i = 0
        while i < len(mutationPool[x]):
            if i == rand:
                j=0
                nodes = random.choice(edgelist)
                while nodes[1] in end:
                    nodes = random.choice(edgelist)
                    if [nodes[0], nodes[1]] in closed:
                        continue
                    if [nodes[1], nodes[0]] in closed:
                        continue
                    if nodes[1] in end:
                        continue
                    mutationPool[x][i] = nodes
                    j+=1
            i+=1
        x += 1
    return mutationPool

#menimpa induk dengan offspring
def regeneration_1(matingPool, popList):
    for i in matingPool:
        popList.append(i)
    
    return popList

#menimpa kromosom yang memiliki fitness terendah
def regeneration(matingPool, popList):
    fitnesses = count_weight(popList)
    population = []
    new_population = []
    for i in range(len(fitnesses)):
        population.append([fitnesses[i], popList[i]])

    ranked_population = sorted(population, reverse = True)

    for i in range(len(ranked_population)):
        new_population.append(ranked_population[i][1])

    for i in range(len(matingPool)):
        new_population.pop()

    for i in range(len(matingPool)):
        new_population.append(matingPool[i])
    
    random.shuffle(new_population)
        
    return new_population

def geneticAlgorithm2(pop_size, generation, probabilitas_OX, Mutation_probability):
    prog = []
    popList = create_population(pop_size, start_node)
    print('bobot untuk generasi pertama: ', counting_weight(random.choice(popList)))
    for i in range(generation):
        selected_pop = roulette_selection(popList, probabilitas_OX)
        matingPool = crossover(selected_pop)
        popList = regeneration_1(popList, matingPool)
        popList = mutation1(Mutation_probability, popList)
        popList = elitis_selection(popList)
        best = counting_weight(popList[0])
        prog.append(best)
        print('Generasi ke-', i+1, 'bobot: ', counting_weight(popList[0]))
        
    popList = elitis_selection(popList)
    print('Hasil optimasi: ', counting_weight(popList[0]))
    plt.plot(prog, linewidth = 1)
    plt.ylabel('Weight')
    plt.xlabel('Generation')
    plt.show()

def geneticAlgorithm(pop_size, generation, probabilitas_OX, Mutation_probability):
    popList = create_population(pop_size, start_node)
    rank = elitis_selection(popList)
    prog = []
    prog.append(counting_weight(rank[0]))
    print('bobot untuk generasi pertama: ', counting_weight(rank[0]))
    i = 0
    while i < (generation):
        #popList = elitis_selection(popList)
        popList = roulette_selection(popList)
        matingPool, sisa = crossover(probabilitas_OX, popList)
        popList = regeneration(matingPool, popList)
        popList = mutation1(Mutation_probability, popList)
        print(count_weight(popList))
        rank = elitis_selection(popList)
        prog.append(counting_weight(rank[0]))
        #print('Generasi ke-', i+1, 'bobot: ', counting_weight(rank[0]))
        print('------------------------------------------')
        #print(count_weight(popList))
        i+=1
    for i in popList:
        print (i)
    print('Hasil optimasi: ', counting_weight(rank[0]))
    plt.plot(prog, linewidth = 1)
    plt.ylabel('Weight')
    plt.xlabel('Generation')
    plt.show()
    #print(count_fitness(rank))
    return popList[0]


popList = geneticAlgorithm(pop_size, generation, probabilitas_OX, Mutation_probability)
print(popList)
make_network_viz(popList)
