import numpy as np
from matrixtools import _unify, _iprod, _jdet, _jeig, _jprod, _furthestPointDist, _closestPointNumber, _allDist
from foldertools import makeMissions, doMissions
from anatools import analizeFile

file_folder =  "dataCohn\dim_6\d6n72v4.npy"
aim_folder = "anadata"
redoExist = False

missions = makeMissions(file_folder, aim_folder, sfx=".npy", sfx2=".csv", redoExist = redoExist)

cos = 0.5
round = 2 # set 0 in case of error
matrixtoolists = [_unify, _iprod, _jprod, _jdet, _jeig, _furthestPointDist, _closestPointNumber, _allDist]

# write any lambda function
analizer = lambda tar_file, matrix: analizeFile(matrixtoolists, matrix, tar_file, round=round, delta=1e-7)

doMissions(missions, np.load, analizer)