import os
import glob

path = "./fine"
ext = "jpg"
file_nums = sum([len(files.extend(glob.glob(dirs + '*.' + e))) for root,dirs,files in os.walk(path)]) 

print(file_nums)
