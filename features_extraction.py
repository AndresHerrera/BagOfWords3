import cv2
import os
import numpy as np
import sys
import time
#from siftdetector import detect_keypoints

def main():
    
    print("Features Extraction from Images folder")

    bagImagesFolder="ImageDataset"
    featuresFolder="Features"

    #bigmatrix = np.empty([0,128])

    start_time = time.time()

    for d in os.listdir(os.path.join(bagImagesFolder)):
    	print ("Read folder: " + d)
    	bigmatrix = np.empty([0,128])
    	for f in os.listdir(os.path.join(bagImagesFolder,d)):
            print ("Read Image: " + f)
            filename=os.path.join(bagImagesFolder,d,f)
            sift=cv2.xfeatures2d.SIFT_create()
            img = cv2.imread(filename)
            (keypoints, descriptors) = sift.detectAndCompute(img,None)
            try:
                print(descriptors.shape)
                if(descriptors.shape[1]==128):
                    bigmatrix=np.concatenate((bigmatrix,descriptors))
                    print(descriptors.shape)
                    print(' Stored in: '+ str(bigmatrix.shape))
            except AttributeError:
                print("shape not found for:" + filename)
            
        print(bigmatrix.shape)
    	np.savetxt(os.path.join(featuresFolder,d+"_features_extracted.csv"), np.asarray(bigmatrix), delimiter=",")

	end_time = time.time()
    elapsed_time = end_time - start_time
    # print total image generation time
    print("Features Extracted total processing time: "+str(elapsed_time) + " seconds")

    #print(bigmatrix.shape)
    #np.savetxt("bag_of_visual_words.csv", np.asarray(bigmatrix), delimiter=",")

if __name__ == "__main__":
    main()
    sys.exit(0)