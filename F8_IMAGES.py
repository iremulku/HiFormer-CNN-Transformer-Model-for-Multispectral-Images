from __future__ import print_function
import os
import torch 
import numpy as np
import scipy.io as sio
import datetime




def get_images(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex):
    input_images=[]
    target_masks=[]    
    gettingfiles=[]
    
    if chindex == 'RGBs':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/RGBs')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/RGBs/{}'.format(b))
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)
    elif chindex == 'NDVIs':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/NDVIs')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/NDVIs/{}'.format(b))            
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)
    elif chindex == 'z3_NDVI_ARVI_SAVI':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/z3_NDVI_ARVI_SAVI')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/z3_NDVI_ARVI_SAVI/{}'.format(b))
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)   
    elif chindex == 'NIRs':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/NIRs')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/NIRs/{}'.format(b))
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)    
    elif chindex == 'NDVIs_wAvg':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/NDVIs_wAvg')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/NDVIs_wAvg/{}'.format(b))
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)             
    elif chindex == 'ARVIs_wAvg':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/ARVIs_wAvg')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            #print(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/ARVIs_wAvg/{}'.format(b),verify_compressed_data_integrity=False)
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)           
    elif chindex == 'SAVIs_wAvg':
        names=os.listdir('D:/Users/akagunduz/deepSegmentation/RIT18/SAVIs_wAvg')
        for b in names[0:trainSetSize]:
            gettingfiles.append(b)
            #print(b)
            a = sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/SAVIs_wAvg/{}'.format(b),verify_compressed_data_integrity=False)
            a = a['inputPatch']
            input_images.append(a)
            c=sio.loadmat('D:/Users/akagunduz/deepSegmentation/RIT18/labels/{}'.format(b))
            c = c['inputPatch']
            target_masks.append(c)   
            
    
    indFile = open("tsind.txt","w");
    d=datetime.datetime.now()
    testnames = open("testnames_{}_{}_{}_{}_{}.txt".format(d.year, d.month, d.day, d.hour, d.minute),"w");
    for say in range(0,len(tsind)):
        indFile.write(str(tsind[say])+'\n')
        testnames.write(str(gettingfiles[tsind[say]])+'\n')
    indFile.close(); 
    testnames.close(); 

    indFile = open("trind.txt","w"); 
    for say in range(0,len(trind)):
        indFile.write(str(trind[say])+'\n')
    indFile.close(); 
    
    indFile = open("vlind.txt","w"); 
    for say in range(0,len(vlind)):
        indFile.write(str(vlind[say])+'\n')
    indFile.close(); 
    
    input_images = np.asarray(input_images, dtype=np.float32)
    target_masks = np.asarray(target_masks, dtype=np.float32)
    lim=224
    input_images = np.reshape(input_images[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3)) 
    input_images = np.moveaxis(input_images,3,1)
    target_masks = np.reshape(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim)) 
    
    trMeanR = input_images[trind,0,:,:].mean()
    trMeanG = input_images[trind,1,:,:].mean()
    trMeanB = input_images[trind,2,:,:].mean()
    
    input_images[:,0,:,:] = input_images[:,0,:,:] - trMeanR
    input_images[:,1,:,:] = input_images[:,1,:,:] - trMeanG
    input_images[:,2,:,:] = input_images[:,2,:,:] - trMeanB
    
    input_images=torch.from_numpy(input_images)
    target_masks=torch.from_numpy(target_masks)
    
    print("image size",input_images.shape,"mask size",target_masks.shape)
    
    print("type image",type(input_images),"type mask",type(target_masks)) 
    
    return input_images, target_masks, trMeanR, trMeanG, trMeanB