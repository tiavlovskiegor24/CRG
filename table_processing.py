import numpy as np
import pandas as pd

def drop_features(df_modified):
    columns_to_drop = ['brcd', 'chrom', 'gene_name',"rep",\
                       "cat","RNA/DNA","pos_expr","RNA","strand",\
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

def encode_one_hot(df):
    barcodes = df.brcd.tolist()

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

    return df_modified,barcodes

def create_matrix(df_modified):
    print "Max value in the data", df_modified.values.ravel().max()
    df_modified.replace(np.nan,1e9,inplace = True)
    feature_names = df_modified.columns.tolist()
    X = df_modified.as_matrix().astype(np.float)
    return X,feature_names

