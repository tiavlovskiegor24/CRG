import numpy as np
import pandas as pd
import file_processing

def table_processing(filename):
    #load feature descriptions
    features = load_feature_names("Features.txt")

    #read the dataframe from file
    df = pd.read_table("Jurkat_BHIVE_mini_expr.txt",comment = "#")

    #import additional features from files
    resolution = "50kb"# select from "10kb","50kb","100kb" and "500kb"
    directory = "/mnt/shared/data/HiC_processing/"
    feature_filenames = {"c_decay":"contacts_decay_Jurkat_",\
                         "gmfpt":"gmfpt_feature_Jurkat_",\
                         "row_sum":"row_sum_Jurkat_"}
    
    df = import_features(df,resolution,directory,feature_filenames)
    

    #enconde categorical features using one hot encoding
    cat_features = ["cat","strand",]
    df,barcodes = encode_one_hot(df,cat_features)

    #drop any features
    drop_features

def train_test_split(df,train_f = 0.8):
    # split into train and test datasets with balanced number of samples for each chromosome
    train_idx = np.array([],dtype=np.int) 
    test_idx = np.array([],dtype=np.int)
    for chrom in df["chrom"].unique():
        idx = np.where(df["chrom"] == chrom)[0]
        n = idx.shape[0]
        s_idx = np.zeros_like(idx,dtype = bool)
        s_idx[np.random.choice(n,int(n*train_f),replace=False)] = True
        train_idx= np.r_[train_idx,idx[s_idx]]
        test_idx = np.r_[test_idx,idx[~s_idx]]

    return train_idx,test_idx


def import_features(df,res = "50kb",directory = None,feature_filenames = None):
    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 50000, '50kb': 50000}
    bins = (df["pos"]/resolution[res]).astype(np.int).values

    for feature in feature_filenames:
        print "Computing feature: %s"%feature
        data = np.loadtxt(directory+feature_filenames[feature]+res,dtype=np.str,delimiter="\t")
        df[feature] = df["pos"]*0
        for chrom in df["chrom"].unique():
            idx = np.where(df["chrom"] == chrom)[0]
            if chrom not in np.unique(data[:,0]):
                print "Alarm ",feature
            df.ix[idx,feature] = data[np.where(data[:,0] == chrom)[0],3][bins[idx]].astype(np.float)
    return df

def drop_features(df_modified,columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = ['brcd','pos', 'chrom', 'gene_name',"rep",\
                       "RNA/DNA","pos_expr","RNA",\
                       "expr","nread","mapq","DNA"]
    
    df_dropped = df_modified[columns_to_drop]
    df_modified.drop(columns_to_drop, inplace=True, axis = 1)
    print df_dropped.head()
    del df_dropped

    
def encode_target_values(df_modified):
    df_modified["RNA/DNA"] = df_modified["RNA"]*1.0/df_modified["DNA"]
    df_modified["pos_expr"] = np.where(df_modified["RNA/DNA"] > 3,1.,0.)
    y = np.array(df_modified["pos_expr"].tolist()).reshape(-1,)\
                                                  .astype(np.float)
    print "Label split: %.2f"%(y.sum()/y.shape)[0]
    print y.shape
    
    return y


def load_feature_names(filename):
    features = {}
    with open(filename,"r") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            feature = [word for word in line[0].split(" ") \
                       if word != ""][1]
            features[feature] = line[1]
    return features


def encode_one_hot(df,cat_features = None):
    if cat_features is None:
        return df
    
    for feature in cat_features:
        f_values = np.unique(df[feature]) 
        print 'There are ' + str(f_values.shape[0]) \
                +' unique values for "%s" feature.\n'%feature
        print f_values 
        dummies = pd.get_dummies(df[feature],prefix = feature,drop_first = True)

        df = pd.concat([df,dummies],axis = 1)
        df.drop(feature,inplace = True,axis = 1)

    '''
    # category of the intergration site
    feature = "cat"
    print 'There are ' + str(np.unique(df[feature]).shape[0]) \
        +' unique values for "%s" feature.\n'%feature
    print np.unique(df[feature])
    dummies = pd.get_dummies(df[feature])

    #print "Unique",dummies.sum(axis = 1).unique()
    dummies.drop("IN",inplace = True,axis = 1)
    df_modified = pd.concat([df,dummies],axis = 1)

    # strand of the integrations site
    feature = "strand"
    print 'There are ' + str(np.unique(df[feature]).shape[0]) \
        +' unique values for "%s" feature.\n'%feature
    print np.unique(df[feature])
    dummies = pd.get_dummies(df[feature])

    #print "Unique",dummies.sum(axis = 1).unique()
    dummies.drop("-",inplace = True,axis = 1)
    dummies.columns = ["Stand +"]
    df_modified = pd.concat([df_modified,dummies],axis = 1)

    # integration site in gene
    df_modified["in_gene"] = np.where(pd.isnull(df_modified["gene_name"])\
                                      ,0.,1.)
    '''
    return df


def create_matrix(df_modified):
    print "Max value in the data", df_modified.values.ravel().max()
    df_modified.replace(np.nan,1e9,inplace = True)
    col_names = df_modified.columns.tolist()
    X = df_modified.as_matrix().astype(np.float)
    return X,col_names

