import numpy
import time
from morph import mmlabel, mmreadgray, mmhmin, mmthreshad, mmlabelflat, mmareaclose, mmareaopen
import glob
import Image
import os

os.system("wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz")
os.system("tar -xzvf BSDS300-images.tgz")

imgs = glob.glob("BSDS300/images/test/*.jpg")

for n,i in enumerate(imgs):

    f = i.split('/')[-1].split('.')[0]
    print n, f

    k = mmreadgray(i)
    size = k.shape
    scale = 4000.0/min(size)
    newsize = (size[0]*scale, size[1]*scale)

    img = (mmhmin(k, 50) > 128).astype(numpy.uint8)*255
    img = mmareaclose(img, 20)
    img = mmareaopen(img, 20)
    pil_img = Image.fromarray(img, "L")
    pil_img = pil_img.resize(newsize)
    pil_img.save( "output/"+f+".png" )
    os.system("convert -compress none %s %s" % 
              ("output/"+f+".png", "output/"+f+".pgm"))

import sys
sys.exit(0)
