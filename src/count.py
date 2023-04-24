import os
import re

f = open("../log/count.txt", "w")

tmp = []

walk_ls = sorted(os.walk("../log"))

for idx, i in enumerate(walk_ls):
    fdir = i[0]
    for fname in i[2]:
        if fname != "log.txt":
            continue
        last = open(os.path.join(fdir, fname)).read().strip().split("\n")[-1]
        if "Test Loss" not in last:
            continue
        roc = last.split(" | ")[3].split(": ")[1] # a text
        f.write(" | ".join([fdir, roc]) + "\n")
        tmp.append(eval(roc))
    if idx == len(walk_ls)-1 or walk_ls[idx+1][0][:-1] != fdir[:-1]:
        if len(tmp) > 0:
            f.write(" | ".join([fdir[:-1]+"_avg", str(sum(tmp) / len(tmp))]) + "\n")
            tmp = []
f.close()