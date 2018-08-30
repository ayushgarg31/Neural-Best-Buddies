import torch
import skimage.io
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(models.vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {3, 8, 17, 26, 35}:
                results.append(x)
        
        return results

    
    
def forward_pass(img, net):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    I = skimage.io.imread(img)
    I = I.astype(float)/255
    I = torch.from_numpy(I.transpose([2, 0, 1]))
    I = I.float()
    I = normalize(I)
    f = net.forward(I.unsqueeze(0))
    for i in range(5):
        f[i] = f[i].permute(0,2,3,1)
    return f
    
    

def visualize(img1,img2,lam):
    im = np.array(Image.open(img1), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    
    im1 = np.array(Image.open(img2), dtype=np.uint8)
    fig1,ax1 = plt.subplots(1)
    ax1.imshow(im1)
        
    for i in [4,3,2]:
        if i == 4:
            n = 18
            m = 84
        elif i == 3:
            n = 6
            m = 36
        else:
            n = 1
            m = 12
        l = lam[i]
        color = np.random.rand(len(l),3)
        a = [x[0][0] for x in l]
        b = [x[0][1] for x in l]
        e = [x[1][0] for x in l]
        d = [x[1][1] for x in l]
        for j in range(len(l)):
            rect = patches.Rectangle((b[j]*(2**(i-1))-n, a[j]*(2**(i-1))-n),m,m,linewidth=1,edgecolor=color[j],facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((d[j]*(2**(i-1))-n, e[j]*(2**(i-1))-n),m,m,linewidth=1,edgecolor=color[j],facecolor='none')
            ax1.add_patch(rect)

    plt.show()
    
    l = lam[1]
    a = [x[0][0] for x in l]
    b = [x[0][1] for x in l]
    e = [x[1][0] for x in l]
    d = [x[1][1] for x in l]

    fig,ax = plt.subplots(1)
    ax.imshow(im)
    color = np.random.rand(len(l))
    plt.scatter(b,a, c = color)
    plt.show()

    fig,ax = plt.subplots(1)
    ax.imshow(im1)
    plt.scatter(d,e, c = color)
    plt.show()

    
    
def neural_best_buddies(img1, img2):
    start = time.time()
    net = Vgg19()
    print ("Initialization time : %s s"% (time.time() - start))
    
    start = time.time()
    fa = forward_pass(img1, net)
    print ("Forward pass for image 1 : %s s"% (time.time() - start))
    
    start = time.time()
    fb = forward_pass(img2, net)
    print ("Forward pass for image 2 : %s s"% (time.time() - start))
    
    start = time.time()
    lam = [[],[],[],[],[],[]]
    lam[5].append([(0,0),(0,0)])
    nx = 3
    n = 14
    for v in range(4,0,-1):
        if (v == 2):
            nx = 2
        for cx in lam[v+1]:
            if (v == 4):
                x1 = fa[4][0].view(14*14,-1)
                x2 = fb[4][0].view(14*14,-1).transpose(0,1)
                f = torch.mm(x1,x2)
            else:
                if (v == 3 or v == 2):
                    n = 7
                else:
                    n = 5

                xa = fa[v][0][cx[0][0]:cx[0][0]+n, cx[0][1]:cx[0][1]+n]
                xb = fb[v][0][cx[1][0]:cx[1][0]+n, cx[1][1]:cx[1][1]+n]
                meana = torch.mean(torch.mean(xa, 0), 0)
                meanb = torch.mean(torch.mean(xb, 0), 0)

                stda = torch.std(torch.std(xa, 0), 0)
                stdb = torch.std(torch.std(xb, 0), 0)

                mean = (torch.add(meana, meanb))/2
                std = (torch.add(stda, stdb))/2

                k = torch.div(std, stda)
                k[k != k] = 0
                k[k == float("Inf")] = 0
                l = torch.div(std, stdb)
                l[l != l] = 0
                l[l == float("Inf")] = 0
                fc = torch.add(torch.mul((xa - meana), k), mean)
                fd = torch.add(torch.mul((xb - meanb), l), mean)

                x1 = fc.view(n*n,-1)
                x2 = fd.view(n*n,-1).transpose(0,1)
                f = torch.mm(x1,x2)

                
            for i in range(1, n-1):
                for j in range(1, n-1):
                    m = -1000000
                    x = 0
                    y = 0
                    for p in range(1, n-1):
                        for q in range(1, n-1):
                            l = 0
                            for a in range(3):
                                for b in range(3):
                                    for c in range(3):
                                        for d in range(3):
                                            l = l + f[(i+a-1)*n+j+b-1][(p+c-1)*n+q+d-1].item()

                            if (l > m):
                                m = l
                                x = p
                                y = q

                    m = -1000000
                    g = 0
                    h = 0
                    for p in range(1, n-1):
                        for q in range(1, n-1):
                            l = 0
                            for a in range(3):
                                for b in range(3):
                                    for c in range(3):
                                        for d in range(3):
                                            l = l + f[(p+a-1)*n+q+b-1][(x+c-1)*n+y+d-1].item()
                            if (l > m):
                                m = l
                                g = p
                                h = q

                    if (g == i and h == j):
                        lam[v].append([(max((i+cx[0][0])*2-nx,0), max((j+cx[0][1])*2-nx,0)),(max((x+cx[1][0])*2-nx,0), max((y+cx[1][1])*2-nx,0))])

    print ("Neural best buddies layer: %s s"% (time.time() - start))
    start = time.time()
    

    visualize(img1, img2, lam)
    return lam
