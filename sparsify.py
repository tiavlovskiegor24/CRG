from scipy.sparse import coo_matrix
def sparsify(X,limit = 0):
    # function takes sparse or dense matrix X and converts it to sparse coo format
    # then it removes it leaves only entries that are equal or below the limit
    # prints the previous and resultant density
    # returns sparese matrix converted to CSR format

    if not isinstance(X, coo_matrix):
        X = coo_matrix(X)
    d1 = (X.count_nonzero()*1./X.shape[0]**2)
    X.row = X.row[(X.data >= limit)]
    X.col = X.col[(X.data >= limit)]
    X.data = X.data[(X.data >= limit)]
    X = X.tocsr()
    print "Density \tbefore:%.3f\n\t\tafter %.3f"%(d1,X.count_nonzero()*1./X.shape[0]**2)
    return X

