import Image, ImageDraw
import numpy

def gen(n):
    d = 4*n + 6
    im = Image.new("L", (d,d))
    draw = ImageDraw.Draw(im)

    r = d/2 - 1
    
    size = 0
    posx = r
    posy = r

    for k in range(d):
        if size % 4 == 0:
            draw.line((posx, posy, posx, posy+size), fill=255)
            posy+=size+1

        elif size % 4 == 1:
            draw.line((posx, posy, posx+size, posy), fill=255)
            posx+=size+1

        elif size % 4 == 2:
            draw.line((posx, posy, posx, posy-size), fill=255)
            posy-=size+1

        elif size % 4 == 3:
            draw.line((posx, posy, posx-size, posy), fill=255)
            posx-=size+1

        size +=1
    
    del draw 
    return im

i = 1000
im = gen(i)
im.save("out.pgm")
