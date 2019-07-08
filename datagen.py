from PIL import Image, ImageDraw, ImageFont
from os.path import isfile, join
from os import listdir
import cv2

size = 400
count = 0

fonts = [f for f in listdir("fonts/") if isfile(join("fonts/", f))]
print(len(fonts))
for f in fonts:
        count +=1
        for i in range (0,10):
                img = Image.new('RGB', (500, 500), color = (0, 0, 0))
                d = ImageDraw.Draw(img)
                font = ImageFont.truetype(join('fonts/', f), size)
                ts = d.textsize(str(size), font = font)
                x = 450 - (ts[0] / 2)
                y = 215 - (ts[1] / 2)
                d.text((x,y),str(i), font=font, fill=(255,255,255))
                img.save('dataset/'+str(i)+'/'+str(count).zfill(3)+'.jpg')
