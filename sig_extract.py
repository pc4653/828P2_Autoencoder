import numpy as np
from sklearn.decomposition import NMF
import scipy.optimize as op
def noise_normalize( mut ):
    mut = np.load('data/all_exomes.npy')
    mut = np.transpose(mut).astype(float)
    new_mut = np.zeros(new_mut.shape)
    for i in range(len(mut)):
        noise = abs(np.random.normal(mut[i].mean()/50, mut[i].var()/50, mut[i].shape)).astype(float)
        new_mut[i] = mut[i] + noise
        new_mut[i] = mut[i]/max(mut[i])
    
    return new_mut


def dim_reduction( M ):
    """
    this reduce all the mutataion types that in total add up to no more than 1 percent of the total
    mutation counts, assuming M is of size # of types * # of observation
    """
    tot_count_per_type = M.sum(axis = 1)
    tot_count = float(tot_count_per_type.sum())
    sorted_index = np.argsort(tot_count_per_type)
    threshold = 0.01
    accu = 0
    for i in range(len(sorted_index)):
        perc = float(tot_count_per_type[sorted_index[i]])/tot_count
        accu = accu + perc
        if accu > threshold:
            break;
            
    return sorted_index[0:i]
    
	
def bootstrap( M ):
    """
    The author normalize the across every observation (column), and uses that as the multinomial 
    distribution that Monte Carlos Simluation draws from
    """
    tot_ct_per_ob = M.sum(axis = 0)
    zero_list = []
    for i in range(len(tot_ct_per_ob)):
        if tot_ct_per_ob[i] <= 0:
            zero_list.append(i)
    M = np.delete(M, zero_list, 1)
    new_M = np.zeros(M.shape)
    tot_ct_per_ob = M.sum(axis = 0)
    for i in range(len(M)):
        for j in range(len(M[0])):
            new_M[i][j] = M[i][j]/tot_ct_per_ob[j]
    new_M = np.transpose(new_M)
    bootstrap = []
    for i in range(len(tot_ct_per_ob)):
        rnd_vec = np.random.multinomial(tot_ct_per_ob[i], new_M[i])
        bootstrap.append(rnd_vec)
            
    bootstrap = np.transpose(np.asarray(bootstrap))        
    return bootstrap		 
			
def normalize(P, E):
    """
    normalize P so that it looks like a distribution
    """
    total = P.sum(axis = 0)
    P = np.transpose(P)
    for i in range(len(P)):
        P[i] = P[i]/total[i]
        E[i] = E[i] * total[i]
    P = np.transpose(P)
    return P, E



def iteration( M, sign_num):
    """
    bootstrap and use NMF (multiplicative update version), random initiation, and 
    normalize P 
    """
    M_bootstrap = bootstrap(M)
    model = NMF(n_components = sign_num, solver = 'mu', max_iter = 10000000, init = 'random')
    #P = np.random.rand(len(M_bootstrap), sign_num)
    #E = np.random.rand(sign_num, len(M_bootstrap[0]))
    P = model.fit_transform(M_bootstrap)
    E = model.components_
    error = model.reconstruction_err_
    P , E = normalize(P, E)
    return P, error

def cos_sim(vec1, vec2):
    
    """
    Compute the similarity of 2 vectors
    
    """
    if len(vec1) != len(vec2):
        print 'dimension does not agree.'
    numerator_sum = 0    
    for i in range(len(vec1)):
        numerator_sum = numerator_sum + vec1[i]*vec2[i]
                                             
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    return numerator_sum/denom
    
    
def avg_sil_width(cluster_vec, centroid):
    
    """
    input a cluster of vectors and the average of that cluster, calculate the avg sil width with
    consine similarity; 1.00 meaning all vectors are exactly the same
    """
    tot_dis = 0
    for i in xrange(len(cluster_vec)):
        cos_dis = cos_sim(cluster_vec[i], centroid)
        tot_dis = tot_dis + cos_dis
    return tot_dis/len(cluster_vec)
    
def opt_exposure(P, M):
    M = np.transpose(M)
    recon_E = []
    error = []
    for i in range(len(M)):
        temp_E, temp_error = op.nnls(P, M[i])
        recon_E.append(temp_E)
        error.append(temp_error)
    recon_E = np.asarray(recon_E)
    error = np.asarray(error)
    return np.transpose(recon_E), error

    
    
    
