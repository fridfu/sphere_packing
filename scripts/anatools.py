import numpy as np
from csv import writer

def verticalAnalizerTool(Matrix, MatFunclist, round = 3, delta = 1e-7):
    results = []
    for MatFunc in MatFunclist:
        eval = MatFunc(Matrix)
        if round:
            try:
                eval = np.round(eval, round)
            except:
                pass
        results.append((MatFunc.__name__, eval))
    return results

def analizeFile(MatFunclist, matrix, tar_file, round = 3, delta = 1e-7):
    results = verticalAnalizerTool(matrix, MatFunclist, round, delta)
    # seperate the filename from .npy and parent folder
    with open(tar_file, 'w', newline='') as csvfile:
        write = writer(csvfile)
        for result in results:
            write.writerow([result[0]])
            eval = result[1]
            onedim = False
            if hasattr(eval, '__iter__'):
                for i in eval:
                    if not hasattr(i, '__iter__'):
                        onedim = True
                        break
                    write.writerow(i)
                if onedim:
                    write.writerow(eval)
            else:
                write.writerow([eval])
            write.writerow([])
    return