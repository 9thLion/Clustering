#! user/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import ClusterPackage as clu
import time
import sklearn.metrics
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

K = np.array([2,3,4,5])
Sil1=[]
Sil2=[]
T1=[]
T2=[]
for a in K:
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
	t1 = time.clock()
	Centroids, Labels = clu.Kmeans(X, k=a, reps=5) #returns the Labels of the clusters as a vector
	t2 = time.clock()
	T1.append(t2-t1)
	S1 = clu.Silhouette(X,Labels)
	plt.subplot(121)
	plt.title('After Kmeans')
	plt.scatter(X[0],X[1], c=Labels)
	#Why do I have to use the transpose of Centroids for plotting when the ones that end up on plot are the rows?
	plt.scatter(Centroids.T[0],Centroids.T[1], marker='x', c='k')

	plt.subplot(122)
	plt.title('After Mix of Gaussians')
	t1 = time.clock()
	Means, Labels = clu.MoG(X, K=a, reps=15) 
	t2 = time.clock()
	T2.append(t2-t1)
	S2 = clu.Silhouette(X,Labels)
	plt.scatter(X[0],X[1], c=Labels)
	plt.scatter(Means.T[0],Means.T[1], marker='x',c='k')
	plt.savefig('Clusters{}.png'.format(a))
	plt.close()
	Sil1.append(S1)
	Sil2.append(S2)

Sil1=np.array(Sil1)
Sil2=np.array(Sil2)
T1=np.array(T1)
T2=np.array(T2)

fig = plt.figure()

ax = fig.add_subplot(121)
ax.set_title('K-means validation')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Coefficient')
ax.set_xticks(K)
ax.scatter(K, Sil1) 

ax = fig.add_subplot(122)
ax.set_title('MoG validation')
ax.set_xlabel('Number of Clusters')
ax.set_xticks(K)
ax.scatter(K, Sil2)
fig.savefig('Validation Test')

fig = plt.figure()

ax = fig.add_subplot(121)
ax.set_title('K-means duration')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Time Elapsed')
ax.set_xticks(K)
ax.scatter(K, T1) 

ax = fig.add_subplot(122)
ax.set_title('MoG duration')
ax.set_xlabel('Number of Clusters')
ax.set_xticks(K)
ax.scatter(K, T2)
fig.savefig('Speed Test')
