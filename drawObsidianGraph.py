import numpy as np
import os
import json

def matrix2ObsidianGraph(matrix, aim_folder, name, cos = 0.5, delta = 0.01):
    os.makedirs(aim_folder + "//" + name, exist_ok=False)
    for i in range(matrix.shape[0]):
        with open(aim_folder + "//" + name + "//" + str(i).zfill(4) + ".md", "w") as f: 
            for j in range(matrix.shape[0]):
                if cos + delta> matrix[i][j] > cos - delta:
                    f.write("[["+str(j).zfill(4)+"]]") # no more than 10000 balls !

def npy2ObsidianGraph(aim_folder, file, cos = 0.5, delta = 0.01):
    matrix = open(file, "r").read()
    matrix = matrix.replace("array(", "")
    matrix = matrix.replace(")", "")
    matrix = matrix.replace("np.float64(", "")
    matrix = np.array(json.loads(matrix))
    mat = matrix @ matrix.T
    matrix2ObsidianGraph(mat, aim_folder, file, cos, delta)


if __name__ == "__main__":
    aim_folder = "graph2obsidian"       # select a folder to save the sub-folder that present the graph
    file = "raw_data2131"  # the .npy file of balls' centres.
    c = 0.5
    d = 0.01 #  c - d < cos(x, y) < c + d then x and y are connected

    npy2ObsidianGraph(aim_folder, file, cos=c, delta=d)
    # You should download the markdown note-taking software Obsidian (https://obsidian.md/) to open the graph.
    # Obsidian => (manage vault) => open folder as vault
    # => Ctrl + P to open command palette => type "Open Graph view" and Enter.

    # Adjust "Node Size"; "center force" and "repel force" to draw better
    # you may manually delete a node file and turn on "Existing files only", to remove that point.