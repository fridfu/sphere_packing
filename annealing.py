
import numpy as np


def force(x1, x2, k, max_dis_allowed=float("inf")):
    r = np.linalg.norm(x1 - x2)
    if r == 0:
        return np.random.randn(*x1.shape) * 0.1
    f = min(pow(1 / r, k), max_dis_allowed, 10)
    return (x1 - x2) / r * f


def force2(x1, x2, k, max_dis_allowed=float("inf")):
    r = np.linalg.norm(x1 - x2)
    if r == 0:
        return np.random.randn(*x1.shape) * 0.1
    if r < 1 - 9 / k:
        f = 10
    elif r < 1:
        f = -k * r + k + 1
    else:
        f = pow(1 / r, k)
    f = min(f, max_dis_allowed)
    return (x1 - x2) / r * f


def move(x, d):
    # move x along direction d but remain on the unit sphere
    # first order approximation
    new_x = x + d
    return new_x / np.linalg.norm(new_x)

def move2(x, d):
    # move x along direction d but remain on the unit sphere
    # exact form
    d = d - x * np.inner(x, d)
    a = np.linalg.norm(d)
    if a == 0:
        return x
    d = d / a
    y = x * np.cos(a) + d * np.sin(a)
    return y / np.linalg.norm(y)


class simulatedAnnealing():
    def __init__(
            self,
            n,
            m,
            initial_state=None,
            lr=1e-2,
            momentum=.0,
            e=1e-10,
            e2=1e-5,
            k=2,
            stop_at=1e+5,
    ):
        """
        dim = n, number of spheres = m
        """
        self.n = n
        self.m = m
        # generate random unit vectors
        if initial_state:
            self.vectors = initial_state
        else:
            random_vectors = np.random.randn(m, n)
            self.vectors = random_vectors / np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]
        self.lr = lr
        self.momentum = momentum
        self.e = e
        self.e2 = e2
        self.k = 2
        self.stop_at = stop_at

        # simulation variables
        self.epoch = 1
        self.min_dis = 2
        self.delta = 1
        self.current_speed = np.zeros_like(self.vectors)

    def display(self):
        print(self.vectors)
        print(f"Iter = {self.epoch}, delta = {self.delta}, min dis = {self.min_dis}")
        print("-" * 24)

    def calc_force(self):
        vectors = self.vectors
        total_force = np.zeros_like(vectors)
        max_dis_allowed = self.delta * 1000 / self.lr
        dis = []
        for i in range(self.m):
            for j in range(i + 1, self.m):
                dis.append(np.linalg.norm(vectors[i] - vectors[j]))
                total_force[i] += force(vectors[i], vectors[j], self.k, max_dis_allowed)
                total_force[j] += force(vectors[j], vectors[i], self.k, max_dis_allowed)
        self.min_dis = min(dis)
        return total_force

    def show_results(self):
        # check whether is a packing
        if self.delta < self.e:
            print("- Equilibrium is achieved!")
        else:
            print("- Equilibrium is not achieved!")
        if self.min_dis > 1 - self.e2:
            print("- The result is a packing!")
        else:
            print("- The result is not a packing!")
        if self.epoch >= self.stop_at:
            print("- Simulation ended at epoch limit!")

    def simulate(self):
        while True:
            # calculate force
            total_force = self.calc_force()
            # move
            new_speed = self.momentum * self.current_speed + self.lr * total_force
            new_vectors = [move2(x, f) for x, f in zip(self.vectors, new_speed)]
            self.delta = max([np.linalg.norm(new_x - x) for new_x, x in zip(new_vectors, self.vectors)])
            # step up k
            if self.delta < self.e2 and self.k < 50:
                print(f"k increase from {self.k} -> {self.k + 1}")
                self.k += 1
            # condition for ending simulation
            if self.delta < self.e or \
                    self.min_dis > 1 - self.e2 or \
                    self.epoch > self.stop_at:
                break
            # journaling
            if self.epoch > 1000 and self.epoch % 1000 == 0 and self.min_dis < .5:
                print("Something is fishy...")
            if self.epoch % 1000 == 0:
                self.display()
            # ready to loop
            self.vectors = new_vectors
            self.current_speed = new_speed
            self.epoch += 1

        self.display()
        self.show_results()


if __name__ == "__main__":
    s = simulatedAnnealing(n=4, m=24, momentum=.9)
    s.simulate()
