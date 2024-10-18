import copy

import numpy as np
from .gradient import GDSphere


def random_rotation_matrix(n):
    """
    Generates a random n-dimensional rotation matrix.
    """
    # Step 1: Create a random n x n matrix with Gaussian entries
    A = np.random.randn(n, n)

    # Step 2: Perform QR decomposition on the matrix A
    Q, R = np.linalg.qr(A)

    # Step 3: Ensure Q is a proper rotation (det(Q) = 1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def random_rotate(parent: GDSphere):
    n = parent.n
    Q = random_rotation_matrix(n)
    parent.vectors = parent.vectors @ Q
    parent.speed = parent.speed @ Q
    return parent


def mix(parent1: GDSphere, parent2: GDSphere):
    # random rotate
    parent1 = random_rotate(parent1)
    parent2 = random_rotate(parent2)

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
        step_times=10,
        population=100,
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

    def explain(self, write_to_file):
        with open(write_to_file, "a") as f:
            f.write("Genetic algorithm starts with the following parameters...")
            f.write(f"dim = {self.n}\n# of balls = {self.m}\n"
                    f"step times = {self.step_times}\npopulation = {self.population}\n"
                    f"p_cross = {self.p_cross}\np_mutate = {self.p_mutate}\n")

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
            new_pool.append(copy.deepcopy(self.pool[i]))
        self.pool = new_pool

    def crossover(self):
        new_pool = []
        for _ in range(self.population):
            if np.random.rand() < self.p_cross:
                i, j = np.random.choice(range(self.population), size=2, replace=False)
                new_pool.append(mix(self.pool[i], self.pool[j]))
            else:
                i = np.random.choice(range(self.population))
                new_pool.append(copy.deepcopy(self.pool[i]))
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
        self.pool[0] = copy.deepcopy(self.best_answer)
        self.epoch += 1

    def display(self, file):
        print("-" * 24)
        print(f"Genetic algorithm... Current epoch = {self.epoch}")
        print("Delta:")
        print([s.delta for s in self.pool])
        print("k:")
        print([s.k for s in self.pool])
        print("Min dis:")
        print([s.min_dis for s in self.pool])
        print(f"End of epoch = {self.epoch}")
        print("-" * 24)

        with open(file, "a") as f:
            f.write("-" * 24)
            f.write(f"\nGenetic algorithm... Current epoch = {self.epoch}\n")
            f.write("Vectors:\n")
            f.write(str([list(s.vectors) for s in self.pool]))
            f.write("\nDelta:\n")
            f.write(str([s.delta for s in self.pool]))
            f.write("\nk:\n")
            f.write(str([s.k for s in self.pool]))
            f.write("\nMin dis:\n")
            f.write(str([s.min_dis for s in self.pool]))
            f.write(f"\nEnd of epoch = {self.epoch}\n")
            f.write("-" * 24 + "\n")



if __name__ == "__main__":
    n, m = 5, 40
    g = GeneticSphere(n=n, m=m)
    file = f"genetic-{n}-{m}"
    g.explain(file)
    for _ in range(1000):
        g.step()
        g.display(file)
        if g.best_answer.min_dis > 1 - 1e-5:
            break
    print("-" * 24)
    print("Final display...")
    g.display(file)

