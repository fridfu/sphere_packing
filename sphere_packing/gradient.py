
import numpy as np


def max_index(a):
    m = max(a)
    for i in range(len(a)):
        if m == a[i]:
            return i
    return 0


def coulomb_force(d, r, k=2, max_dis_allowed=float("inf")):
    if r == 0:
        return np.random.randn(*d.shape) * 0.1
    f = min(pow(1 / r, k), max_dis_allowed, 10)
    return d / r * f


def elastic_force(d, r, k=2):
    if r >= 1:
        return 0
    if r == 0:
        return np.random.randn(*d.shape) * 0.1
    f = pow(1 - r, k - 1)
    return d / r * f


def elastic_potential(r):
    if r >= 1:
        return 0
    return pow(1 - r, 2)


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
    # move x along direction d on the sphere
    # d is already perpendicular to x
    # d = d - x * np.inner(x, d)
    a = np.linalg.norm(d)
    if a == 0:
        return x
    d = d / a
    y = x * np.cos(a) + d * np.sin(a)
    return y / np.linalg.norm(y)


class GDSphere():
    def __init__(
            self,
            n,
            m,
            initial_state=None,
            k=2,
            epoch=0,
            min_dis=2,
            delta=1,
            lr=1e-2,
            momentum=.0,
            e=1e-10,
            e2=1e-5,
    ):
        """
        n: dim,
        m: number of spheres,
        e: threshold for termination,
        e2: threshold for increasing k,
        k: penalty level (on the exponential),
        """
        self.n = n
        self.m = m
        # generate random unit vectors
        if initial_state:
            self.vectors = np.array(initial_state)
        else:
            random_vectors = np.random.randn(m, n)
            self.vectors = random_vectors / np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]
        self.lr = lr
        self.momentum = momentum
        self.e = e
        self.e2 = e2
        self.k = k

        # simulation variables
        self.epoch = epoch
        self.min_dis = min_dis
        self.delta = delta
        self.speed = np.zeros_like(self.vectors)
        self.best_vectors = self.vectors
        self.best_min_dis = 2
        self.most_crowded_index = -1

    def display(self):
        print("-" * 24)
        print(f"Iter = {self.epoch}")
        print("Speed:")
        print([list(v) for v in self.speed])
        print("Positions:")
        print([list(v) for v in self.vectors])
        print(f"Iter = {self.epoch}, delta = {self.delta}, k = {self.k}, min dis = {self.min_dis}")
        print("-" * 24)

    def calc_force(self):
        # the force to be used
        f = elastic_force
        u = elastic_potential

        vectors = self.vectors
        total_force = np.zeros_like(vectors)
        potential = np.zeros(self.m)

        # # TODO: this is a wierd/good method?
        # max_dis_allowed = self.delta * 1000 / self.lr

        dis = []
        for i in range(self.m):
            for j in range(i + 1, self.m):
                d = vectors[i] - vectors[j]
                r = np.linalg.norm(d)
                dis.append(r)
                # calculate force
                # total_force[i] += f(vectors[i], vectors[j], self.k, max_dis_allowed)
                # total_force[j] += f(vectors[j], vectors[i], self.k, max_dis_allowed)
                total_force[i] += f(d, r)
                total_force[j] -= total_force[i]
                # calculate potential
                v = u(r)
                potential[i] += v
                potential[j] += v
        self.min_dis = min(dis)
        return total_force, potential

    def step_n(self, n, display=False):
        for _ in range(n):
            self.step(display=display)

    def step(self, display=True):
        # calculate force and potential
        total_force, potential = self.calc_force()

        # move
        new_speed = self.momentum * self.speed + self.lr * total_force
        new_speed1 = [d - x * np.inner(x, d) for x, d in zip(self.vectors, new_speed)]
        new_vectors = [move(x, f) for x, f in zip(self.vectors, new_speed1)]
        self.delta = max([np.linalg.norm(new_x - x) for new_x, x in zip(new_vectors, self.vectors)])

        # get ready for next step
        self.vectors = new_vectors
        self.speed = np.array(new_speed1)
        self.epoch += 1
        if self.min_dis > self.best_min_dis:
            self.best_min_dis = self.min_dis
            self.best_vectors = self.vectors

        # if we sense that we are close to a local minimum
        if self.delta < self.e:
            # # increase the penalty exponential
            # if self.k < 50:
            #     self.k += 1
            # pick out the most crowded sphere
            # and move it to a random place
            # a sphere will not be moved in two
            # consecutive times
            i = max_index(potential)
            if i != self.most_crowded_index:
                print(f"Sphere {i} is so crowded with potential {potential[i]}!")
                print("Moving it to a different place...")
                d = np.random.randn(self.n)
                self.vectors[i] = (d / np.linalg.norm(d))
                self.speed[i] = np.zeros(self.n)
                self.most_crowded_index = i

        # if get into a wierd position...
        # if self.epoch > 1000 and self.epoch % 1000 == 0 \
        #         and (self.min_dis < .5 or self.min_dis < self.best_min_dis):
        #     if display:
        #         print(f"Something is fishy... min dis = {self.min_dis}")

    def mutate(self):
        i = np.random.randint(self.m)
        random_vector = np.random.randn(self.n)
        self.vectors[i] = random_vector / np.linalg.norm(random_vector)

        self.k = 2
        self.speed = np.zeros_like(self.vectors)
