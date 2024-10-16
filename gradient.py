
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
        dim = n, number of spheres = m
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
        self.current_speed = np.zeros_like(self.vectors)
        self.best_vectors = self.vectors
        self.best_min_dis = 2

    def display(self):
        print(self.vectors)
        print(f"Iter = {self.epoch}, delta = {self.delta}, k = {self.k}, min dis = {self.min_dis}")
        print("-" * 24)

    def calc_force(self):
        vectors = self.vectors
        total_force = np.zeros_like(vectors)
        # TODO: this is a wierd/good method?
        max_dis_allowed = self.delta * 1000 / self.lr
        dis = []
        for i in range(self.m):
            for j in range(i + 1, self.m):
                dis.append(np.linalg.norm(vectors[i] - vectors[j]))
                total_force[i] += force(vectors[i], vectors[j], self.k, max_dis_allowed)
                total_force[j] += force(vectors[j], vectors[i], self.k, max_dis_allowed)
        self.min_dis = min(dis)
        return total_force

    def step_n(self, n, display=False):
        for _ in range(n):
            self.step(display=display)

    def step(self, display=True):
        # calculate force
        total_force = self.calc_force()
        # move
        new_speed = self.momentum * self.current_speed + self.lr * total_force
        new_vectors = [move2(x, f) for x, f in zip(self.vectors, new_speed)]
        self.delta = max([np.linalg.norm(new_x - x) for new_x, x in zip(new_vectors, self.vectors)])
        # step up k
        if self.delta < self.e2 and self.k < 50:
            self.k += 1
        # get ready for next step
        self.vectors = new_vectors
        self.current_speed = new_speed
        self.epoch += 1
        if self.min_dis > self.best_min_dis:
            self.best_min_dis = self.min_dis
            self.best_vectors = self.vectors
        # if get into a wierd position...
        if self.epoch > 1000 and self.epoch % 1000 == 0 \
                and (self.min_dis < .5 or self.min_dis < self.best_min_dis):
            if display:
                print(f"Something is fishy... min dis = {self.min_dis}")
            # self.mutate()

    def mutate(self):
        i = np.random.randint(self.m)
        random_vector = np.random.randn(self.n)
        self.vectors[i] = random_vector / np.linalg.norm(random_vector)

        self.k = 2
        self.current_speed = np.zeros_like(self.vectors)

    def update_min_dis(self):
        ...


if __name__ == "__main__":
    s = GDSphere(n=5, m=40, momentum=.9, initial_state=[[-0.70672705, -0.5784003, -0.40086329, -0.04328639, 0.0585225], [-0.76044838, -0.12581917, 0.36665379, 0.12860119, 0.50489061], [-0.80279703, 0.23388183, -0.21208386, 0.48161483, -0.15454388], [-0.7756535, 0.30990728, -0.22057002, -0.47885695, 0.15608976], [0.19999605, -0.82048348, -0.1088311, -0.21326259, -0.47904416], [0.15933522, -0.63992776, -0.6963627, 0.05013694, 0.27869342], [0.12497634, -0.18847197, 0.9629759, 0.02677157, 0.14429111], [0.103743, 0.42000265, 0.14294126, -0.8215195, -0.34279537], [0.08604042, -0.02244785, 0.35080066, 0.28465209, 0.88769658], [0.06529529, 0.57203635, -0.45037924, -0.54413456, 0.41180949], [0.06583815, 0.31383614, 0.15479185, 0.51974105, -0.77658286], [0.02826196, 0.46831089, -0.43880113, 0.76628692, -0.01200409], [0.0360624, -0.16169066, 0.59003391, -0.25849607, -0.74672311], [-0.06583815, -0.31383614, -0.15479185, -0.51974105, 0.77658286], [-0.103743, -0.42000265, -0.14294126, 0.8215195, 0.34279537], [-0.12497634, 0.18847197, -0.9629759, -0.02677157, -0.14429111], [-0.12961964, -0.87865658, 0.29914646, 0.03634371, 0.3469177], [-0.15036477, -0.28417238, -0.50203344, -0.79244294, -0.12896939], [-0.1873981, -0.38789784, -0.49045534, 0.51797854, -0.55278297], [-0.19999605, 0.82048348, 0.1088311, 0.21326259, 0.47904416], [-0.22365958, 0.33330753, 0.54512992, -0.55792779, 0.48003378], [-0.2606929, 0.22958207, 0.55670803, 0.75249368, 0.05622019], [-0.28143803, 0.82406627, -0.24447188, -0.07629297, -0.4196669], [-0.44712007, 0.10710035, -0.53535933, 0.21854681, 0.67397391], [-0.54294196, -0.41260203, 0.39107996, -0.61240249, -0.08362568], [-0.57008549, -0.48862748, 0.39956612, 0.3480693, -0.39425932], [-0.5852906, -0.05290103, -0.18765769, -0.25938884, -0.74306017], [-0.63901194, 0.3996801, 0.57985939, -0.08750126, -0.29669206], [0.84994349, -0.00914492, 0.32292974, -0.10943016, -0.40156427], [0.72850704, -0.53464419, 0.10972414, 0.10667228, 0.40001839], [0.71330192, -0.09891774, -0.47749967, -0.50078586, 0.05121754], [0.6861584, -0.17494319, -0.46901351, 0.45968593, -0.25941609], [0.65958059, 0.35366339, 0.29001741, -0.32889827, 0.49758565], [0.63243706, 0.27763794, 0.29850357, 0.63157351, 0.18695202], [0.61723194, 0.71336439, -0.28872024, 0.02411537, -0.16184883], [0.4957955, 0.18786512, -0.50192584, 0.24021782, 0.63973383], [0.38279571, -0.41372058, 0.38412655, -0.72633766, 0.08475328], [0.34576239, -0.51744604, 0.39570466, 0.58408382, -0.3390603], [0.32501726, 0.07703816, -0.40547525, -0.24470283, -0.81494739], [0.25172246, 0.69451807, 0.64168812, -0.01018769, -0.20594423]])
    for _ in range(100):
        s.step_n(1000, display=True)
        s.display()
        if s.delta < s.e:
            print("Equilibrium is achieved!")
            break
        if s.min_dis > 1 - s.e2:
            print("Eureka! A solution is found!")
            break
