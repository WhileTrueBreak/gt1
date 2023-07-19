import pathlib

def getFiles(path):
    l = []
    for item in pathlib.Path(path).iterdir():
        if not item.is_file(): continue
        l.append(str(item))
    return l