
import numpy as np
from sphere_packing import GeneticSphere
from datetime import datetime

n, m = 5, 40
g = GeneticSphere(n=n, m=m)
file = f"../data/genetic-{n}-{m}-{datetime.now().strftime("%Y-%m-%d-%H:%M")}"
g.explain(file)
for _ in range(1000):
    g.step()
    g.display(file)
    if g.best_answer.min_dis > 1 - 1e-5:
        break
print("-" * 24)
print("Final display...")
g.display(file)