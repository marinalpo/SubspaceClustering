import numpy as np

class RwHoptCond:
  # conditions for reweighted heuristic 
  def __init__(self, maxIter, eigThres, corner):
    self.maxIter = maxIter # number of max iterations
    self.eigThres = eigThres  # threshold on the eigenvalue fraction for stopping procedure
    self.corner = corner  # rank-1 on corner (1) or on full matrix (0)
    

def createNormalVectors(D, Ns):
  # Generates a matrix of Normal Vectors from a random distribution
  # Inputs: - D: dimension
  #         - Ns: number of Subspaces
  # Outputs: - Normals: DxNs matrix with normal vectors
  normals = np.random.randn(D, Ns)
  normals = (np.matmul(normals,np.diag(np.sign(normals[0,:]))))/(np.sqrt(np.diag(np.matmul(np.transpose(normals),normals))))
  b = np.argsort(-normals[0,:])
  normals = normals[:,b]
  return normals


def generatePoints(normals, num_points, noise_b, sq_side):
  # Generates points from a set of subspaces, by sampling from a square of D-1 dimensions and rotating about the normal.
  # Inputs: - Normals: DxS matrix with the S normals in D dimensions
  #         - num_points: Sx1 vector with the number of points to be sampled in each subspace
  #         - noise_b: noise bound.
  #         - sq_side: size of the square from which we sample.
  # Outputs: - X: DxN matrix with the N points
  #          - ss_ind: index of the subspace.
    N = np.sum(num_points)
    [D, S] = normals.shape
    X = np.zeros([D, N])
    ss_ind = np.zeros([N, 1])
    k = 0
    for ss in range(S):
      X_tmp = np.vstack((2 * (np.random.uniform(0, 1, [1, num_points[ss]]) - 0.5) * noise_b,
                         2 * (np.random.uniform(0, 1, [(D - 1), num_points[ss]]) - 0.5) * sq_side))
      SVD = np.linalg.svd(
        np.eye(D) - (1 / np.sqrt(np.dot(normals[:, ss], normals[:, ss]))) * np.outer(normals[:, ss], normals[:, ss]))
      U = np.fliplr(SVD[0])
      X_tmp = np.matmul(U, X_tmp)
      X[:, k:(k + num_points[ss])] = X_tmp
      ss_ind[k:(k + num_points[ss])] = ss * np.ones((num_points[ss], 1))
      k = k + num_points[ss]
    return X, ss_ind
