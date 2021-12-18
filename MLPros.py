import numpy as np
from sklearn.decomposition import PCA

def PCASpect(Spectra,npc,Standardize,**kwargs):

    X = Spectra.copy()
    nrec, nwl = Spectra.shape


    if Standardize == True:
        meanspect = kwargs['meanspect']
        stdspect = kwargs['stdspect']
        mean2d = np.stack([meanspect] * nrec, axis=0)
        std2d = np.stack([stdspect] * nrec, axis=0)
        X = (X - mean2d) / std2d
        # meanspect = np.mean(Spectra, axis=0)
        # print(Spectra-Spectra1)
        # exit()

    pca = PCA(n_components=npc, svd_solver='full')
    pca.fit(X)

    if Standardize==True:
        return [meanspect,stdspect,pca]
    else:
        return pca

def PCAapply(pca, data, Standardize,**kwargs):

    X=data.copy()
    nrec, nwl = X.shape

    if Standardize==True:
        meanspect=kwargs['meanspect']
        stdspect=kwargs['stdspect']
        mean2d = np.stack([meanspect] * nrec, axis=0)
        std2d = np.stack([stdspect] * nrec, axis=0)
        X = (X - mean2d) / std2d

    return pca.transform(X)

def PCAreconstruct(pcs, pc_score, Standardize,**kwargs):

    recondata=np.matmul(pc_score,pcs)
    if Standardize==True:
        nrec, npc = pc_score.shape
        meanspect=kwargs['meanspect']
        stdspect=kwargs['stdspect']
        mean2d = np.stack([meanspect] * nrec, axis=0)
        std2d = np.stack([stdspect] * nrec, axis=0)
        recondata = recondata*std2d+mean2d


    return recondata