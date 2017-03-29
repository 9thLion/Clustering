import numpy as np 
import random as ran

def Kmeans(X, k=2):
	def Euc(x,y):
		return (np.linalg.norm(x-y)) #an alias, basically
	centroids=[]
	#Take some random data points as centroids like so:
	for x in range(k): 
		centroids.append(ran.choice(X.T))
	centroids = np.array(centroids)
	while True:
		Clusters = {x:[] for x in range(k)} #the index of the centroid within the centroids array will match the key
		Color=[]
		for x in X.T: #data points
			centroid = np.array(centroids[0]) #starting with the first
			Eucprevious = Euc(x,centroid)
			counter=index=0 #The counter at the end of each loop will match the index of the centroid
			for c in centroids[1:]:
				c = np.array(c)
				if Euc(x,c) < Eucprevious:
					Eucprevious = Euc(x,c)
					centroid = c
					counter+=1 #first add one, so that the counter will be the same as the index
					index = counter 
				else:
					counter+=1
			Color.append(index) #to visualize the process
			Clusters[index].append(x)

		centroids_new=[] #empty it
		for key in Clusters:
			#stack the values of each cluster as columns of a new array
			OldCluster = np.column_stack(Clusters[key])
			#and now compute the new centroids
			centroids_new.append(np.mean(OldCluster,1))
		centroids_new = np.array(centroids_new)
		if Euc(centroids_new, centroids) < 0.00001:
			break
		else:
			print('One more loop done and the new centroids are', centroids_new)
			centroids = centroids_new
	return(Color)