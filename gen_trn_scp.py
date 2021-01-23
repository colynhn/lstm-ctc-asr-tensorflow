import os
import sys
from tqdm import tqdm

fdir = sys.argv[1]
files = os.listdir(fdir)
for f in tqdm(files):
	if f[-3:] == "wav":
		with open(sys.argv[2]+, "a") as wavf:
			wavf.write(fdir+f+"\n")
		with open(sys.argv[3], "a") as trnf:
			trnf.write(fdir+f+".trn"+"\n")
print("all done.")
        
"""
first = trf.readline()
line = first.strip("\n ").split(" ")
s = ""
for l in line:
    s=s+l
    s = list(s)
    print(s)
    exit()
    """
