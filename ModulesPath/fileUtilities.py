import os


def listDirectory(folderpath, match_folder=''):

    folderpath = os.path.expanduser(folderpath)
    list_folders = [dI for dI in os.listdir(
        folderpath) if os.path.isdir(os.path.join(folderpath, dI))]

    folder_name = [i for i in list_folders if match_folder in i]

    sort_folders = sorted(folder_name)

    return sort_folders
