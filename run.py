import numpy
import time
from morph import mmlabel, mmreadgray, mmhmin, mmthreshad, mmlabelflat, mmareaclose, mmareaopen
import glob
import Image
import subprocess
import os

import csv

OUTPUT_SIMPLE = "image_set/output/"

imgs = glob.glob(OUTPUT_SIMPLE+"*.png")

#imgs = ["image_set/output/101085.png",]

Writer = csv.writer(open('results/results.csv', 'w'), delimiter=' ',quotechar='|',
                    quoting=csv.QUOTE_MINIMAL)

def run(img):

    d = {"cc" : 0,
         "gold" : 0,
         "uf" : 0,
         "uf_hybrid" : 0,
         "lequiv" : 0,
        }

    p = subprocess.Popen(["./bin/ccl_test", "%s.pgm"%img], stdout=subprocess.PIPE)
    lines = p.stdout.readlines()

    for l in lines:
        header, content = l.split(":")
        d[header] = content

    numbercc = int(d["cc"])
    gold_serial = float(d["gold"])
    uf_gpu = float(d["uf"].split()[0])
    uf_hybrid = float(d["uf_hybrid"].split()[0])
    lequiv_gpu = float(d["lequiv"].split()[0])

    return [numbercc, gold_serial, uf_gpu, uf_hybrid, lequiv_gpu]

l_uf = numpy.zeros(len(imgs))
l_uf_hybrid = numpy.zeros(len(imgs))
l_lequiv = numpy.zeros(len(imgs))
l_stephano = numpy.zeros(len(imgs))

for n,i in enumerate(imgs):

    RESULTS = []

    f = os.path.split(OUTPUT_SIMPLE+i)[1].split(".")[0]
    res = run(OUTPUT_SIMPLE+f)

    k = (mmreadgray(OUTPUT_SIMPLE+f+".png") > 0)

    v1 = []
    for j in range(10):
        t0 = time.time()
        lbl = mmlabel(k)
        t1 = time.time()
        v1.append(1000*(t1-t0))

    RESULTS = res+[min(v1)]

    l_uf[n]        = RESULTS[2]
    l_uf_hybrid[n] = RESULTS[3]
    l_lequiv[n]    = RESULTS[4]
    l_stephano[n]  = RESULTS[5]

    print(f)
    print("\tcc:\t%d\n\tgold:\t%f\n\tuf:\t%f\n\tuf_hybrid:\t%f\n\tlequiv:\t%f\n\tstephano:\t%f\n" % tuple(RESULTS) )
    Writer.writerow(RESULTS)

print(" == Finals Results == \n")
s1 = l_stephano/l_uf
s2 = l_stephano/l_uf_hybrid
s3 = l_stephano/l_lequiv
print("union-find (gpu)\n\tmean:%4.2f\n\tstd:%4.2f\n\tmax:%4.2f\n\tmin:%4.2f"     % (numpy.mean(s1), numpy.std(s1), numpy.max(s1), numpy.min(s1)) )
print("union-find (gpu+cpu)\n\tmean:%4.2f\n\tstd:%4.2f\n\tmax:%4.2f\n\tmin:%4.2f" % (numpy.mean(s2), numpy.std(s2), numpy.max(s2), numpy.min(s2)) )
print("Label Equivalence\n\tmean:%4.2f\n\tstd:%4.2f\n\tmax:%4.2f\n\tmin:%4.2f"    % (numpy.mean(s3), numpy.std(s3), numpy.max(s3), numpy.min(s3)) )

import sys
sys.exit(0)
