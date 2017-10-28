from __future__ import division
import cv2
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid



cluster_size = 150


def main():
    aciertos=0
    testFolder="TestDataset"
    dirsf= os.listdir(testFolder)
    for ff in dirsf:
        print('Reading '+ff )

        sift=cv2.xfeatures2d.SIFT_create()
        imgo = cv2.imread(os.path.join(testFolder,ff))
        img = cv2.resize(imgo, (100, 50))

        labelO=ff.split("_")[0]
         
        (keypoints, descriptors) = sift.detectAndCompute(img,None)
        
        print(descriptors.shape)

        featuresFolder="VisualDictionary"

        md=[]
        lb=[]

        dirs= os.listdir(featuresFolder)
        for f in dirs:
            file = np.array(list(csv.reader(open(os.path.join(featuresFolder,f), "rb"), delimiter=","))).astype("float")
            
            X = np.array(file)
            y = np.array(descriptors)
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(y)
            distances, indices = nbrs.kneighbors(X)
            
            hist2, bin_edges2 = np.histogram(np.array(descriptors).flatten(), bins = range(cluster_size+1), normed=True ) 
            #normed=True, , density=True
            #print hist2

            dst = distance.euclidean(distances[:,1],hist2)
            #print dst

            label=f.split("_")[0]
            print label
            print dst
            md.append( dst )
            lb.append( label )

        pos=md.index(min(md))
        print('pos->'+str(pos))
        print('predicted->'+lb[pos])
        print('real->'+labelO)
        if(labelO==lb[pos]):
            aciertos+=1
        print md
        print lb
    
    testsize=len(dirsf)
    print 'Test dataset size: '+ str(testsize)
    print 'Aciertos: ' + str(aciertos)
    print 'Desempeno: '+str((int(aciertos))/(int(testsize))*100)+' %'

     
if __name__ == "__main__":
    main()
    sys.exit(0)