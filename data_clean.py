from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from skimage import io
import selectivesearch
import shutil
import numpy as np
def intersect(rec1,rec2):
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])     
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])       
    # computing the sum_area     
    sum_area = S_rec1 + S_rec2       
    # find the each edge of intersect rectangle     
    left_line = max(rec1[1], rec2[1])     
    right_line = min(rec1[3], rec2[3])     
    top_line = max(rec1[0], rec2[0])     
    bottom_line = min(rec1[2], rec2[2])       
    # judge if there is an intersect     
    if left_line >= right_line or top_line >= bottom_line:         
        return 0     
    else:         
        intersect = (right_line - left_line) * (bottom_line - top_line)         
    return (intersect / (sum_area - intersect))*1.0
k = 1
s = 0
for i in range(1,8):
    with open("FDDB-folds/FDDB-fold-0"+str(i)+"-ellipseList.txt","r") as f:
        path = f.readline()
        while path:
            #img = Image.open(path+'.jpg')
            img = io.imread(path[:-1]+'.jpg')
            minx = 0
            maxx = img.shape[0]
            miny = 0
            maxy = img.shape[1]
            try:
                img_lbl, regions = selectivesearch.selective_search(img, scale = 1000, sigma=0.9, min_size=100)
            except:
                num = int(f.readline())
                for i in range(num):
                    f.readline()
                path = f.readline()
                continue
            shutil.copyfile(path[:-1]+".jpg","data/train/"+str(k)+".jpg")
            r = []
            for each in regions:
                [x1,y1,x2,y2] = each["rect"]
                if ((x2<30) or (y2<30)):
                    continue
                x2 = x1 + x2
                y2 = y1 + y2
                flag = True
                for p in r:
                    if (intersect(p,[x1,y1,x2,y2])>0.9):
                        flag = False
                if flag:
                    r.append([x1,y1,x2,y2])
            #plt.imshow(img)
            num = int(f.readline())
            bnd_box = []
            for i in range(num):
                para = f.readline().split(" ")[:-2]
                cx = float(para[-2])
                cy = float(para[-1])
                theta = float(para[2])
                l = float(para[0])
                s = float(para[1])
                x1 = cx - abs(l*math.cos(theta)) - abs(s*math.sin(theta))
                x1 = max(x1,minx)
                y1 = cy - abs(l*math.sin(theta)) - abs(s*math.cos(theta))
                y1 = max(y1,miny)
                x2 = cx + abs(l*math.cos(theta)) + abs(s*math.sin(theta))
                x2 = min(x2,maxx)
                y2 = cy + abs(l*math.sin(theta)) + abs(s*math.cos(theta))
                y2 = min(y2,maxy)
                bnd_box.append([x1,y1,x2,y2])
                #print(x1,y1,x2,y2)
                #plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1])
            #plt.show()
            np.savez("data/train/"+str(k)+".npz",bndbox = bnd_box, regions = r)
            path = f.readline()
            k += 1
    print(k)