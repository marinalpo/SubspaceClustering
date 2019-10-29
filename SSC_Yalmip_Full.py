def SSC_CVXPY_Full(Xp, eps, Ns, RwHopt, delta):
  # Subspace Clustering using CVXPY, Naive implementation
  # Inputs: - Xp: DxNp matrix, each column is a point
  #         - eps: allowed distance from the subspace
  #         - Ns: number of subspaces
  #         - RwHopt: conditions for reweighted heuristic contained on an object that includes:
  #                   - maxIter: number of max iterations
  #                   - eigThres: threshold on the eigenvalue fraction for stopping procedure
  #                   - corner: rank-1 on corner (1) or on full matrix (0)
  #         - delta: noise factor on the identity at first iteration
  # Outputs: - R: tensor of length Ns, where each item is a (1+D)x(1+D) matrix with the subspace coordinates
  #          - S: NsxNp matrix, with labels for each point and subspace
  #          - runtime: runtime of the algorithm (excluding solution extraction)
  #          - rankness: ???
  return R, S, runtime, rank1ness