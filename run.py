import numpy
import time
from morph import mmlabel, mmreadgray, mmhmin, mmthreshad, mmlabelflat, mmareaclose, mmareaopen
import glob
import Image
import subprocess
import os

import csv

OUTPUT_SIMPLE = "image_set/output_simple/"

imgs = glob.glob(OUTPUT_SIMPLE+"*.png")

#imgs = ["image_set/output/101085.png",]

Writer = csv.writer(open('results.csv', 'w'), delimiter=' ',quotechar='|',
                    quoting=csv.QUOTE_MINIMAL)

def run(img):

    d = {"cc" : 0,
         "gold" : 0, 
         "uf_total" : 0,
         "lequiv_total" : 0,
        }

    p = subprocess.Popen(["./test", "%s.pgm"%img], stdout=subprocess.PIPE)
    lines = p.stdout.readlines()

    for l in lines:
        header, content = l.split(":")
        d[header] = content

    numbercc = int(d["cc"])
    gold_serial = float(d["gold"])
    uf_gpu = float(d["uf_total"].split()[0])
    lequiv_gpu = float(d["lequiv_total"].split()[0])

    return [numbercc, gold_serial, uf_gpu, lequiv_gpu]

for n,i in enumerate(imgs):
    
    RESULTS = []

    f = os.path.split(OUTPUT_SIMPLE+i)[1].split(".")[0]
    print OUTPUT_SIMPLE+f
    res = run(OUTPUT_SIMPLE+f)

    k = (mmreadgray(OUTPUT_SIMPLE+f+".png") > 0)

    v1 = []
    for i in range(10):
        t0 = time.time()
        lbl = mmlabel(k)
        t1 = time.time()
        v1.append(1000*(t1-t0))

    RESULTS = res+[min(v1)]

    print RESULTS
    Writer.writerow(RESULTS)

import sys
sys.exit(0)
