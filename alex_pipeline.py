import numpy as np
import csv
import sig_extract
import time
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import scipy.io as io
import sys

#hyperparameters
sign_num=int(sys.argv[3])
iteration = 500
remove_perc = 0.1

file_name = sys.argv[1]


signatures = np.load(sys.argv[2])


M = np.load(file_name)
if len(M) != 96:
    M = np.transpose(M)
    
elim_type = sig_extract.dim_reduction(M)
#elim_type = []
print 'eliminating categories: ' 
print sorted(elim_type)
M_dot = np.delete(M, elim_type, axis=0)
M_dot = M_dot.astype(float)

#for every iteration of NMF, get the error and P and put them into a list
P_list = []
error_list = []
since = time.time()
for i in xrange(iteration):
    P, error = sig_extract.iteration(M_dot, sign_num)
    P_list.append(P)
    error_list.append(error)


time_elapsed = time.time() - since
print 'completed in ' + str(time_elapsed) + 'seconds.'

#remove iterations that have high error
remove_iter = iteration*remove_perc
sorted_index = np.argsort(error_list)
temp = sorted_index[len(sorted_index)-int(remove_iter):len(sorted_index)]
error_list = np.delete(error_list, temp)
P_list = np.delete(P_list, temp, axis = 0)

print "the average Frobenius reconstruction error is: "
print error_list.mean()
	

	

#putting all verisions of mutation signatures into one matrix
cluster_vec = np.transpose(P_list[0])
for i in range(1,len(P_list)):
    cluster_vec = np.vstack([cluster_vec, np.transpose(P_list[i])])


#use kmeans to find clusters of N    
kmeans = KMeans(n_clusters=sign_num, random_state=0).fit(cluster_vec)

print "the forbenius reconstruction error for the set of estimated P is: "
P_est = np.transpose(kmeans.cluster_centers_)
E_est, E_error = sig_extract.opt_exposure(P_est, M_dot)
np.save(sys.argv[1] + '_exposure.npy', E_est)
error = M_dot - np.dot(P_est, E_est)
est_error = np.linalg.norm(error,'fro')
print est_error    
print "the recon error from finding optimal exposure is: " 
print np.linalg.norm(E_error)
#seperating clusters into its own list and calculate the average silhoutte width
#of each cluster
cluster = {}
for i in range(sign_num):
    cluster[i] = []

for i in range(sign_num):
    for j in range(len(kmeans.labels_)):
        if kmeans.labels_[j] == i:
            cluster[i].append(cluster_vec[j])

cluster_sil = []

for i in range(sign_num):
        cluster_sil.append(sig_extract.avg_sil_width(cluster[i], kmeans.cluster_centers_[i]))

print "the average silhoutte width for each cluster is:"
print cluster_sil
print "average silhoutte width for all is: "
print sum(cluster_sil)/sign_num    

#putting the eliminated signatures back into the extracted signatures for 
#comparison's sake
elim_type = np.sort(elim_type)
result = []
for i in range(sign_num):
    a = kmeans.cluster_centers_[i]
    for j in elim_type:
        a = np.insert(a, j, 0)

    result.append(a)
result = np.asarray(result)
np.save('extracted_sig.npy',result)    
	
	
best_match = []
for i in range(len(result)):
    best_match.append([0,0])
all_val = []
for i in range(len(signatures)):
    val = []
    for j in range(len(result)):
		val.append(sig_extract.cos_sim(signatures[i], result[j]))
    all_val.append(val)
    index = np.argsort(val)    
    print "" + str(i) + " signature has the highest similarity with " + str(index[len(index)-1]) + " signatures with " + str(val[index[len(index)-1]]) 
    if best_match[index[len(index)-1]][0] < val[index[len(index)-1]]:
        best_match[index[len(index)-1]][0] = val[index[len(index)-1]]
        best_match[index[len(index)-1]][1] = i
print best_match
import matplotlib.pyplot as plt
plt.imshow(all_val, cmap='hot', interpolation='nearest')
plt.show(block=True)

