#! usr/bin/python3.5
import numpy as np
import ClusterPackage as clu
import PCApackage as pac
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os

#os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
#os.system('gunzip GDS6248.soft.gz')
#os.system('tail -n +141 GDS6248.soft > GDS6248.softer') #getting rid of the redundant lines
#os.system('rm GDS6248.soft')
#os.system('head -n -1 GDS6248.softer > GDS6248.soft') #one last redundant line
#os.system('rm GDS6248.softer')

#In the following loop I'm keeping the float values while skipping the strings by setting the ValueError exception
temp = []
with open('GDS6248.soft') as f:
	for l in f:
		temp2=[]
		for x in l.split()[2:]:
			try:
				temp2.append(float(x))
			except ValueError: 
				pass
		temp.append(temp2)

X=np.array(temp)
TrueLabels = [2 for x in range(3)] + [0 for x in range(24)] + [1 for x in range(24)]
#because baseline has the lowest effect, ill give it the label 2, as to not cause the accuracy test to fail terribly for the 2 clusters
TL=np.array(TrueLabels)

Y=pac.PCA(X,k=3, F=False)[0]
K=np.array([2,3,4])
Sil=[]
Fs=[]
for a in K:
	Means, Labels = clu.MoG(Y,K=a, reps=10)
	S = clu.Silhouette(Y,Labels) 
	fscore = accuracy_score(TL, Labels) #since we know the true labels we can check the accuracy of each clustering process
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(Y[0],Y[1], c=Labels)
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.scatter(Means.T[0],Means.T[1], marker='x',c='k')
	plt.show()

	Fs.append(fscore)
	Sil.append(S)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Accuracy Test')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Accuracy score')
ax.set_xticks(K)
ax.scatter(K, Fs) 

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Validation Test')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette coefficient')
ax.set_xticks(K)
ax.scatter(K, Fs) 

plt.show()
