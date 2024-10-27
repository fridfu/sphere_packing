import os

def transfer_filename(filepath, sfx = ".txt", sfx2 = ".npy"): # if you wish the sfx2 could be like "thing.npy"
    return filepath.split("\\")[-1].replace(sfx, sfx2)

def folder2processlist(folder_path, sfx = ".txt"): # avoid file starts with "#"
    process_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(sfx) and not file_name.startswith("#"):
            file_path = os.path.join(folder_path, file_name)
            process_list.append(file_path)
    return process_list

def makeMissions(folderORfile_path, aim_folder, sfx = ".txt", sfx2 = ".npy", redoExist = False):
    if folderORfile_path.endswith(sfx):
        afp = transfer_filename(folderORfile_path, sfx=sfx, sfx2=sfx2)
        process_list = [(folderORfile_path, aim_folder + "\\" + afp)]
        print("single ", afp)
    else:
        process_list = []
        if redoExist:
            for filepath in folder2processlist(folderORfile_path, sfx=sfx):
                afp = transfer_filename(filepath, sfx=sfx, sfx2=sfx2)
                process_list.append((filepath, aim_folder + "\\" + afp))
                print("write2 ", afp)
        else:
            alreadyexist = os.listdir(aim_folder)
            for filepath in folder2processlist(folderORfile_path, sfx=sfx):
                afp = transfer_filename(filepath, sfx=sfx, sfx2=sfx2)
                if afp in alreadyexist: print("exists ", afp) ; continue
                process_list.append((filepath, aim_folder + "\\" + afp))
                print("append ", afp)
    return process_list

def doesMission(mission, reader, writer):
    medium = reader(mission[0])
    writer(mission[1], medium)

def doMissions(missions, reader, writer):
    for mission in missions:
        doesMission(mission, reader, writer)