import numpy as np
import pandas as pd
from feature_file_processing import read_feature_file, full_array,ravel_feature_data
from myplot import myplot
import sys
        
def train_test_split(df,train_f = 0.9,chrom_sort = None):
    '''
    by default chrom_sort is passed and the dataset is split
    into train and test datasets with balanced number of samples for each chromosome
    ''' 
    if chrom_sort is not None:
        train_idx = np.array([],dtype=np.int) 
        test_idx = np.array([],dtype=np.int)

        for chrom,idx in chrom_sort.iteritems():
            n = idx.shape[0]
            s_idx = np.zeros_like(idx,dtype = bool)
            s_idx[np.random.choice(n,int(n*train_f),replace=False)] = True
            train_idx= np.r_[train_idx,idx[s_idx]]
            test_idx = np.r_[test_idx,idx[~s_idx]]
    else:
        n = df.shape[0]
        s_idx = np.zeros(n,dtype = bool)
        s_idx[np.random.choice(n,int(n*train_f),replace=False)] = True
        train_idx = np.arange(n)[s_indx]
        test_idx = np.arange(n)[~s_idx]
        
    return train_idx,test_idx



def import_features(df,map_col = "bin",res = "",directory = None,feature_filenames = None):
    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 500000, '50kb': 50000,"":None,"5kb":5000}

    if map_col == "bin":
        bins = df[map_col].values.astype(np.int)
    elif map_col == "pos":
        bins = (df[map_col]/resolution[res]).astype(np.int).values
    else:
        print "INvalid mapping column name"
        return

    target_indices = {}
    for i,(chrom,b) in enumerate(zip(df["chrom"].values,bins)):
        if (chrom,b) in target_indices:
            target_indices[(chrom,b)].append(i)
        else:
            target_indices[(chrom,b)] = [i]


    for feature in feature_filenames:
        #print "Computing feature: %s"%feature
        print "\tImporting feature: {}\n\t\t from {}".format(feature,directory+feature_filenames[feature]+res)
        #data = np.loadtxt(directory+feature_filenames[feature]+res,dtype=np.str,delimiter="\t")
        data = read_feature_file(directory+feature_filenames[feature]+res)
        data = full_array(data,res = resolution[res])
 
        feature_vector = np.ones(df.shape[0])*np.nan
        encountered = set()
        if isinstance(data,dict):
            for chrom,table in data.iteritems():
                for sample in table:
                    ch,start,end,value = sample
                    assert ch == chrom
                    b = int(start)/resolution[res]
                    
                    if (chrom,b) in encountered:
                        print (chrom,b),"already encountered"
                        
                    if (chrom,b) in target_indices:
                        t_idx = target_indices[(chrom,b)]
                        feature_vector[t_idx] = value
                        encountered.add((chrom,b))
                    else:
                        pass
                        #print chrom,b

        df[feature] = feature_vector
        #print np.isnan(df[feature].values).sum()
        
                
        '''
                
            for chrom in df["chrom"].unique():
                idx = np.where(df["chrom"] == chrom)[0]
                if chrom not in data:
                    print "Alarm ",feature,chrom
                    continue
                try:
                    df.ix[idx,feature] = data[chrom]["value"][bins[idx]]
                except:
                    print chrom,bins,idx,data[chrom]["value"]
        else:
            for chrom in df["chrom"].unique():
                idx = np.where(df["chrom"] == chrom)[0]
                if chrom not in np.unique(data[:,0]):
                    print "Alarm ",feature
                df.ix[idx,feature] = data[np.where(data[:,0] == chrom)[0],3][bins[idx]].astype(np.float)
        print np.isnan(df[feature].values).sum()

        '''
    return df



def drop_features(df_modified,columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = ['brcd','pos', 'gene_name',"rep",\
                           "expr","nread","mapq"]
    
    
    df_modified.drop(columns_to_drop, inplace=True, axis = 1)

    return df_modified

    

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
        print '\n\tThere are {} unique values for "{}" feature.'.format(f_values.shape,feature)
        print "\t",f_values 
        dummies = pd.get_dummies(df[feature],prefix = feature+"_oh_",drop_first = False)

        df = pd.concat([df,dummies],axis = 1)

    return df




