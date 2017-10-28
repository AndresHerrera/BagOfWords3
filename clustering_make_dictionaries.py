import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import sys
import csv
import os

def main():
    
	featuresFolder="Features"
	visualDictionaryFolder="VisualDictionary"
	
	dirs= os.listdir(featuresFolder)
	
	for f in dirs:
   		print('Reading '+f )
		file = np.array(list(csv.reader(open(os.path.join(featuresFolder,f), "rb"), delimiter=","))).astype("float")

		filename, file_extension = os.path.splitext(f)
		print(filename)
		label=filename.split("_")[0]

		print(file.shape)
	
		# #Define number of clusters
		K = 150
		X= file

		# #This function creates the classifier	
		kmeans = KMeans(n_clusters=K,init='random')
		# #Original
		kmeans.fit(X)
	
		centroids = kmeans.cluster_centers_
		labels = kmeans.labels_
		n_clusters = len(centroids)

		print labels
		print labels.shape
		print centroids
		print centroids.shape
		print 'N CLusters:'+str(n_clusters)

		#Save dictionary
		savetofilenamec= os.path.join(visualDictionaryFolder,label+"_dictionary_K_"+str(K)+"_centroids.csv")
		#savetofilenamel= os.path.join(visualDictionaryFolder,label+"_dictionary_K_"+str(K)+"_labels.csv")
		np.savetxt(savetofilenamec,np.asarray(centroids), delimiter=",")
		#np.savetxt(savetofilenamel,np.asarray(labels), delimiter=",")

	sys.exit(0)

if __name__ == "__main__":
    main()
