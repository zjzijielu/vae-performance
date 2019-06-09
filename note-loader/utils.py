import numpy as np
import os, fnmatch

def find(pattern, path):
    result = []
    files = os.listdir(path)
    for i in range(len(files)):
        if fnmatch.fnmatch(files[i], pattern):
            result.append(files[i])
    result.sort()
    return result