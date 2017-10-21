# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:46:25 2017

@author: ehsanamid
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:17:14 2017

@author: ehsanamid
"""

from sklearn.neighbors import NearestNeighbors as knn
#from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pylab as plot
    
def generate_triplets(X, k=20):
    num_extra = np.maximum(k+10, 30) # look up more neighbors
    n = X.shape[0]
    nbrs = knn(n_neighbors= num_extra + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    #    sig = distances[:,10]
    sig = np.mean(distances[:, 10:15], axis=1) # scale parameter
    P = np.exp(-distances**2/np.reshape(sig[indices.flatten()],[n, num_extra + 1])/sig[:,np.newaxis])
    sort_indices = np.argsort(-P, axis = 1) # actual neighbors
    triplets = np.zeros([n * k**2, 3])
    weights = np.zeros(n * k**2)
    cnt = 0
    for i in xrange(n):
        for j in xrange(k):
            sim = indices[i,sort_indices[i,j+1]]
            p_sim = P[i,sort_indices[i,j+1]]
            rem = indices[i,sort_indices[i,:j+2]].tolist()
            l = 0
            while (l < k):
                out = np.random.choice(n)
                if out not in rem:
                    triplets[cnt,:] = [i, sim, out]
                    p_out = np.exp(-np.sum((X[i,:]-X[out,:])**2)/(sig[i] * sig[out]))
                    if p_out < 1e-20:
                        p_out = 1e-20
                    weights[cnt] = p_sim/p_out
                    rem.append(out)
                    l += 1
                    cnt += 1
        if ((i+1) % 500) == 0:
            pass
            #print 'Genareted triplets %d out of %d' % (i+1, n)
    weights /= np.max(weights)
    weights += 0.01
    weights = np.log(1 + 50 * weights)
    weights /= np.max(weights)
    return (triplets.astype(int), weights.flatten())
        
def tete_grad(Y, triplets, weights):
    n, dim = Y.shape
    grad = np.zeros([n, dim])
    y_ij = Y[triplets[:,0],:] - Y[triplets[:,1],:]
    y_ik = Y[triplets[:,0],:] - Y[triplets[:,2],:]
    d_ij = 1 + np.sum(y_ij**2,axis=-1)
    d_ik = 1 + np.sum(y_ik**2,axis=-1)
    num_viol = np.sum(d_ij > d_ik)
    denom = (d_ij + d_ik)**2
    loss = weights.dot(d_ij/(d_ij + d_ik))
    gs = 2 * y_ij * (d_ik/denom * weights)[:,np.newaxis]
    go = 2 * y_ik * (d_ij/denom * weights)[:,np.newaxis]
    for i in range(dim):
        grad[:,i] += np.bincount(triplets[:,0], weights= gs[:,i] - go[:,i])
        grad[:,i] += np.bincount(triplets[:,1], weights = -gs[:,i])
        grad[:,i] += np.bincount(triplets[:,2], weights = go[:,i])
    return (loss, grad, num_viol)
    

def tete(X, num_dims = 2, num_neighbs = 20):
    n, dim = X.shape
    X -= np.min(X)
    X /= np.max(X)
    X -= np.mean(X,axis=0)
    if dim > 50:
#        pca = PCA(n_components=50)
#        pca.fit(X)
#        X = np.dot(X, pca.components_.transpose())
        X = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
    Y = np.random.normal(size=[n, 2]) * 0.0001
    C = np.inf
    best_C = np.inf
    best_Y = Y
    tol = 1e-7
    num_iter = 2000
    eta = 500.0 # learning rate
    
    triplets, weights = generate_triplets(X, num_neighbs)
    num_triplets = float(triplets.shape[0])
    
    for itr in range(num_iter):
        old_C = C
        C, grad, num_viol = tete_grad(Y, triplets, weights)
        
        # maintain best answer
        if C < best_C:
            best_C = C
            best_Y = Y
            
        # update Y
        Y -= (eta/num_triplets * n) * grad;
        
        # update the learning rate
        if old_C > C + tol:
            eta = eta * 1.01
        else:
            eta = eta * 0.5
        
        if (itr+1) % 100 == 0:
            pass
            # print 'Iteration: %4d, Loss: %3.3f, Num viol: %0.4f' % (itr+1, C, float(num_viol)/num_triplets)
    #np.savetxt('result_tete_outlier_10000.txt',best_Y)
    return best_Y


if __name__ == "__main__":
    print "Run Y = tete.tete(X, num_dims, num_neighbs) to perform t-ETE on your dataset."
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tete(X, 2, 20)
    plot.scatter(Y[:,0], Y[:,1], 10, labels)
    plot.show();

