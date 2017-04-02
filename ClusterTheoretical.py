#! user/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import ClusterPackage as clu
#Crafting the Artificial Dataset
D=2
N1=220
N2=280

mean = np.ones(D)
def varID(scale):
	return(scale*np.identity(D)) 
X1 = np.random.multivariate_normal(mean, varID(0.5), N1).T #DxN matrix
X2 = np.random.multivariate_normal(-mean, varID(0.75), N2).T
X = np.concatenate((X1,X2), axis=1)

print(X.shape)
#plt.hist(X1.T)
#plt.hist(X2.T)
#plt.show()
#print(X.shape)
#clu.Kmeans(X)
Centroids, Labels = clu.Kmeans(X, reps=5) #returns the Labels of the clusters as a list

plt.subplot(221)
plt.title('After Kmeans')
plt.scatter(X[0],X[1], c=Labels)
#Why do I have to use the transpose of Centroids for plotting when the ones that end up one plot are the rows?
plt.scatter(Centroids.T[0],Centroids.T[1])

plt.subplot(222)
plt.title('After Mix of Gaussians')
Means, Labels = clu.MoG(X, reps=3) #returns the Labels of the clusters as a list
plt.scatter(X[0],X[1], c=Labels)
#Why do I have to use the transpose of Centroids for plotting when the ones that end up one plot are the rows?
plt.scatter(Centroids.T[0],Centroids.T[1])
#Since we know the correct clusters, let's test the effectiveness of this algorithm
#by separating the data
plt.subplot(223)
plt.title('True Labels')
plt.scatter(X1[0],X1[1])
plt.scatter(X2[0],X2[1], c='g')

plt.show()


