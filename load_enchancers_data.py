from scipy.sparse import lil_matrix

def load_enhancers_data(filename):
    '''
    function to load enhancers(on non-enhancers) data set from file 
into sparse scipy matrix
    '''
    all_indices = []
    all_values = []
    width = 0
    with open(filename,"r") as f:
        for i,line in enumerate(f):
            line = line.split("\t")
            non_zeros = ",".join(line[1:])
            if non_zeros:
                exec("indices,values = zip(*[{}])".format(non_zeros))
            else:
                print line
                all_indices.append([])
                all_values.append([])
                continue

            width = max(width,max(indices)+1)
            all_indices.append(indices)
            all_values.append(values)
    del indices,values

    matrix = lil_matrix((len(all_indices),width))
    matrix.data = all_values
    matrix.rows = all_indices
    #matrix = matrix.tocsr()
    
    return matrix
