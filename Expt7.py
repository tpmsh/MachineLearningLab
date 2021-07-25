import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# X contains training data samples from a synthetic dataset
#x = [Nvisits,purchaseAmount]
X = [[10,20],[9,0],[6,5],[7,25],[11,0],[12,10],[8,11],
    [4,150],[5,120],[7,100],[5,200],[6,120],[4,220],[6,300],
    [7,404],[5,388],[6,225],[8,350],[8,236],[4,167],[9,400],
    [9,699],[5,553],[8,450],[9,817],[10,1010],[11,825],[11,700]]

# Number of clusters k (you may find best k by using elbow point)
k = 4
model = KMeans(n_clusters=k,random_state=0) 

# Fitting the input data
kmeans = model.fit(X) #model is trained and stored in kmeans

# Centroid values
Xarr = np.array(X)
centroids = np.around(kmeans.cluster_centers_, decimals=2)
print(f"Cluster Centers are: {centroids}")

# Predicting the cluster labels
labels = kmeans.predict(X)
print(f"Cluster Labels are: {labels}")

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
plt.xlabel('Number of Visits')
plt.ylabel('Purchase Amount')

for i in range(k):
    #plot all points labelled i during each iteration with different color
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    #plot centroid of that cluster
    ax.scatter(centroids[i, 0], centroids[i, 1], marker='*', s=200, c=colors[i])
           
# predicting class of input pattern [4,200]
xnew = np.array([[4,200],[9,500]])
newlabels = kmeans.predict(xnew)

for i in range(len(xnew)):
    print(f"Pattern {xnew[i]} belongs to cluster #{newlabels[i]}")
ax.scatter(xnew[:, 0], xnew[:, 1], s=100, c=[colors[i] for i in newlabels])

# Cluster Analysis:Give One possible interpretation of the four clusters in your own words
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Moderate')
green_patch = mpatches.Patch(color='green', label='Basic')
blue_patch = mpatches.Patch(color='blue', label='Prime')
yellow_patch = mpatches.Patch(color='yellow', label='Window Shopper')

plt.legend(handles=[red_patch, green_patch, blue_patch, yellow_patch])