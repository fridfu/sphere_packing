
import numpy as np
from gradient import GDSphere


def mix(parent1: GDSphere, parent2: GDSphere):
    # take vectors from parent1 whose x-axis is lower than $cut$
    cut = np.random.rand() * 2 - 1
    vectors = [v for v in parent1.vectors if v[0] < cut]

    # take vectors from parent2 with the highest x-axis to make up the number
    vectors2 = sorted(parent2.vectors, key=lambda x: x[0], reverse=True)
    vectors += vectors2[:parent1.m - len(vectors)]
    return GDSphere(
        parent1.n,
        parent1.m,
        initial_state=vectors,
        k=min(parent1.k, parent2.k),
        epoch=parent1.epoch,
        delta=max(parent1.delta, parent2.delta),
    )


class GeneticSphere():
    def __init__(
        self,
        n,
        m,
        step_times=500,
        population=20,
        p_cross=0.5,
        p_mutate=0.05,
    ):
        self.n = n
        self.m = m
        self.step_times = step_times
        self.population = population
        self.p_cross = p_cross
        self.p_mutate = p_mutate

        self.pool = [GDSphere(n, m) for _ in range(population)]
        self.best_answer = self.pool[0]

        self.epoch = 0

    def evaluation(self):
        for _, s in enumerate(self.pool):
            s.step_n(self.step_times)
            if s.min_dis > self.best_answer.min_dis:
                self.best_answer = s

    def selection(self):
        # TODO: optimize the choice
        prob = [pow(s.min_dis, 2) for s in self.pool]

        prob = [p / sum(prob) for p in prob]
        new_index = np.random.choice(range(self.population), size=self.population, replace=True, p=prob)
        new_pool = []
        for i in new_index:
            new_pool.append(self.pool[i])
        self.pool = new_pool

    def crossover(self):
        new_pool = []
        for _ in range(self.population):
            if np.random.rand() < self.p_cross:
                i, j = np.random.choice(range(self.population), size=2, replace=False)
                new_pool.append(mix(self.pool[i], self.pool[j]))
            else:
                new_pool.append(self.pool[np.random.choice(range(self.population))])
        self.pool = new_pool

    def mutation(self):
        for i in range(self.population):
            if np.random.rand() < self.p_mutate:
                self.pool[i].mutate()

    def step(self):
        # the evaluation process runs GD for every candidate
        self.evaluation()
        self.selection()
        self.crossover()
        self.mutation()
        # theoretically, only by adding the best answer up to date
        # can ensure convergence of genetic algorithm
        self.pool[0] = self.best_answer
        self.epoch += 1

    def display(self):
        print("-" * 24)
        print(f"Genetic algorithm... Current epoch = {self.epoch}")
        print("Vectors:")
        print([s.vectors for s in self.pool])
        print("Delta:")
        print([s.delta for s in self.pool])
        print("Min dis:")
        print([s.min_dis for s in self.pool])
        # TODO:print how crossover happens and how mutation happens



if __name__ == "__main__":
    g = GeneticSphere(n=5, m=40)
    for _ in range(100):
        g.step()
        g.display()
