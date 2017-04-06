#! user/bin/python3.5

import numpy as np
import matplotlib.pyplot as plt
import ClusterPackage as clu
import time
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
	S1 = clu.Silhouette(X, Labels)
	plt.subplot(121)
	plt.title('After Kmeans')
	plt.scatter(X[0],X[1], c=Labels)
	#Why do I have to use the transpose of Centroids for plotting when the ones that end up on plot are the rows?
	plt.scatter(Centroids.T[0],Centroids.T[1], marker='x', c='k', alpha=0.7)

	plt.subplot(122)
	plt.title('After Mix of Gaussians')
	t1 = time.clock()
	Means, Labels = clu.MoG(X, K=a, reps=15) 
	t2 = time.clock()
	T2.append(t2-t1)
	S2 = clu.Silhouette(X, Labels)
	plt.scatter(X[0],X[1], c=Labels)
	plt.scatter(Means.T[0],Means.T[1], marker='x',c='k', alpha=0.7)
	plt.savefig('Clusters{}.png'.format(a))
	plt.close()
	Sil1.append(S1)
	Sil2.append(S2)

Sil1=np.array(Sil1)
Sil2=np.array(Sil2)
T1=np.array(T1)
T2=np.array(T2)


plt.subplot(121)
plt.title('K-means validation')
plt.scatter(K, Sil1) 
plt.subplot(122)
plt.title('MoG validation')
plt.scatter(K, Sil2)
plt.savefig('Validation Test')
plt.close()

plt.subplot(121)
plt.title('K-means speed')
plt.scatter(K, T1)
plt.subplot(122)
plt.title('MoG speed')
plt.scatter(K, T2)
plt.savefig('Speed Test')
plt.close()