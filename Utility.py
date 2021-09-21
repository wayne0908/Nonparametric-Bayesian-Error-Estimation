import numpy as np 
import os 
import pdb
import math
import multiprocessing as mp
import time
import scipy.io as sio
import scipy.integrate as integrate
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn import mixture 
from joblib import  Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import multivariate_normal


def GetBER(args, Mean):
    """
    numerically compute BER
    Mean: center of the second component
    """
    X = np.zeros(args.FeatLen); X[0] = 0; X[1:] = np.inf
    BER = multivariate_normal.cdf(X, mean = Mean)
    return BER

# true point nearest neighbour error
# Mean0: mean of gaussian distribution for class 0 
# Mean1: mean of gaussian distribution for class 1
def GetPointNN(Mean0, Mean1, x):

    px_0 = multivariate_normal.pdf(x, mean = Mean0)
    px_1 = multivariate_normal.pdf(x, mean = Mean1)
    px = 0.5 * (px_0 + px_1) 
    PNNErr = 1/2 * px_0 * px_1/px # original point NN error has px**2 in the divisor.
                                # One px has been absorbed by mutiplying px for subsequent 
                                # integration
    return PNNErr

# True nearest neighbour error 
def GetRealDpBounds(args):
    Mean0 = -args.Sep / 2; Mean1 = args.Sep / 2;
    Result = integrate.quad(lambda x: GetPointNN(Mean0, Mean1, x), -10, 10)
    Up = 1 - 2 * Result[0]
    lb = 1/2 - 1/2 * np.sqrt(Up); ub = 1/2 - 1/2 * Up
    return lb, ub
def GetData(Args):
    """
    Args: parse(). option parameters
    """   
    Mean0 = np.zeros(Args.FeatLen); Mean1 = np.zeros(Args.FeatLen);
    Mean0[0] = Mean0[0] - Args.Sep / 2; Mean1[0] = Mean1[0] + Args.Sep / 2 
    Cov0 = np.diag(np.ones(Args.FeatLen)); Cov1 = np.diag(np.ones(Args.FeatLen) +Args.Del)
    BER = GetBER(Args, Mean1) # numerically compute BER
    True_lb, True_ub = GetRealDpBounds(Args) 

    print('Creating synthetic dataset of two component gaussian. Sample Size: %d, Speration:%.2f, Delta:%2f, Dimension: %d, BER:%.4f, lowerbound:%.4f, upperbound:%.4f'%
         (Args.S, Args.Sep, Args.Del, Args.FeatLen, BER, True_lb, True_ub))
    StatsPath = os.getcwd() + '/Stats/%s/Data/D%d/Sep%.2f/Delta%.2f/Size%d/'%(Args.DataType, Args.FeatLen, Args.Sep, Args.Del, Args.S)
   
    S1 = int(Args.S/2); 
    mn0 = multivariate_normal(Mean0, Cov0); 
    mn1 = multivariate_normal(Mean1,Cov1);

    Feat0 = mn0.rvs(size=S1, random_state=Args.Trial).reshape(-1, Args.FeatLen); 
    Feat1 = mn1.rvs(size=S1, random_state=Args.Trial + 1).reshape(-1, Args.FeatLen)

    Data0 = np.concatenate((Feat0, np.zeros((S1, 1))), 1); Data1 = np.concatenate((Feat1, np.ones((S1, 1))), 1); 
    Data = np.concatenate((Data0, Data1), 0); Data = np.random.RandomState(Args.Trial-1).permutation(Data); 

    if not os.path.exists(StatsPath):
        os.makedirs(StatsPath)
    if Args.SaveData:
        np.save(StatsPath + 'SynSep%.2fDel%.2fSize%d.npy'%(Args.Sep, Args.Del, Args.S), Data); 
    return Data, BER, True_lb, True_ub

def BEREstimate(Data):
    """
    Estimate Bayes error rate upper-bound and lower bound
    """
    NumCores = mp.cpu_count()

    Feat = Data[:, :-1]
    Lab = Data[:, -1]

    """
    DP estimation
    """
    GraphMatrix = np.triu(pairwise_distances(Feat, n_jobs = NumCores), 0)
    csr = minimum_spanning_tree(csr_matrix(GraphMatrix)) 
    minimum_tree = csr.toarray()
    ConnectMatrix = np.triu(pairwise_distances(Lab.reshape((-1,1)), n_jobs = NumCores), 0)
    Dp = 1 - 2 * np.sum(minimum_tree.astype(bool) * ConnectMatrix) / (len(Feat)) # Dp divergence
  
    """
    Bounds BER by DP
    """
    BERLower = 0.5 - 0.5 * np.sqrt(Dp); BerUpper = 0.5 - 0.5 * Dp
    print('Dp: %.4f, BER lower-bound: %.4f, BER upper-bound: %.4f'%(Dp,BERLower, BerUpper))
    return BERLower, BerUpper, Dp

def BinaryKernelBayesLowerbound(Data, Sigma = 0.5):
    """
    Graph construction and connectional construction
    """
    NumCores = mp.cpu_count()
    # # weight matrix
    TimeStart = time.clock()
    GraphMatrix = pairwise_distances(Data[:, :-1], n_jobs = NumCores)
    ConnectMatrix = pairwise_distances(Data[:, -1].reshape((-1, 1)), n_jobs = NumCores)

    """
    Lowerbound estimation
    """
    N = len(GraphMatrix); print('data size: %d'%N)
    GraphMask = GraphMatrix < np.mean(GraphMatrix) * Sigma
    PointBayesLowerbound = []
    """
    ParallelProcessing
    """
    def NonparEstimate(i):
        kernelData = Data[GraphMask[i], :]
        KernelGraphMatrix = GraphMatrix[GraphMask[i], :][:, GraphMask[i]]
        KernelConnectMatrix = ConnectMatrix[GraphMask[i], :][:, GraphMask[i]]       
        # cluster KNN error calculation by minimum spanning tree
        Tcsr = minimum_spanning_tree(csr_matrix(KernelGraphMatrix)) 
        minimum_tree = Tcsr.toarray()
        # pdb.set_trace()
        NumFR = np.sum(minimum_tree.astype(bool) * KernelConnectMatrix) # Binary case

        # Binary case
        N0 = np.sum(kernelData[:, -1] == 0)
        N1 = np.sum(kernelData[:, -1] == 1)
        # print("Point %d Finished"%i)
        return 0.5 - 0.5*math.sqrt(max(0, 1 - 2 * NumFR / (N0+N1))) 

    
    PointBayesLowerbound = Parallel(n_jobs=NumCores)(delayed(NonparEstimate)(i) for i in range(N))
    TimeElapsed = (time.clock()-TimeStart)
    BER = np.mean(PointBayesLowerbound)
    # print(TimeElapsed)
    print("point-wise estimate of BER:%.4f"%BER)
    return PointBayesLowerbound, BER