
# coding: utf-8

# In[195]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[199]:

get_ipython().magic(u'run ~/CRG/scripts/ml_process.py')
get_ipython().magic(u'run ~/CRG/scripts/load_enchancers_data.py')
get_ipython().magic(u'run ~/CRG/scripts/ml_inputs.py')


# In[200]:

enhancers = load_enhancers_data("enh-sparse.feat")
enhancers


# In[201]:

non_enhancers = load_enhancers_data("noenh-sparse.feat")
non_enhancers


# In[202]:

from sklearn.model_selection import train_test_split


# In[203]:

train_enchancers,test_enchancers,_,_ = train_test_split(enhancers,np.ones(enhancers.shape[0]),test_size = 0.2)
train_non_enchancers,test_non_enchancers,_,_ = train_test_split(non_enhancers,np.ones(non_enhancers.shape[0]),test_size = 0.2)
del enhancers,non_enhancers


# In[204]:

train_enchancers = train_enchancers.tolil()
test_enchancers = test_enchancers.tolil()
train_non_enchancers = train_non_enchancers.tolil()
test_non_enchancers = test_non_enchancers.tolil()


# In[205]:

n_train = train_enchancers.shape[0]+train_non_enchancers.shape[0]
n_test = test_enchancers.shape[0]+test_non_enchancers.shape[0]
m = train_enchancers.shape[1]
n_train,n_test,m


# In[206]:

train_samples = lil_matrix((n_train,m))
test_samples = lil_matrix((n_test,m))


# In[207]:

train_samples.rows = train_enchancers.rows.tolist()+train_non_enchancers.rows.tolist()
train_samples.data = train_enchancers.data.tolist()+train_non_enchancers.data.tolist()
test_samples.rows = test_enchancers.rows.tolist()+test_non_enchancers.rows.tolist()
test_samples.data = test_enchancers.data.tolist()+test_non_enchancers.data.tolist()


# In[208]:

train_targets = np.array([1]*train_enchancers.shape[0] + [0]*train_non_enchancers.shape[0])
test_targets = np.array([1]*test_enchancers.shape[0] + [0]*test_non_enchancers.shape[0])


# In[209]:

train_samples = train_samples.tocsr().log1p()
test_samples = test_samples.tocsc().log1p()


# In[210]:

train_percentile = np.percentile(train_samples.data,99)
train_samples.data[train_samples.data > train_percentile] = train_percentile


# In[211]:

test_percentile = np.percentile(test_samples.data,99)
test_samples.data[test_samples.data > percentile] = test_percentile


# In[212]:

train_samples.data = train_samples.data / train_samples.data.max()
test_samples.data = test_samples.data / test_samples.data.max()


# In[214]:

train_samples.data.max()


# In[216]:

plt.hist(train_samples.data,bins=100)
plt.show()


# In[217]:

ML_inputs = {
    "train_samples":train_samples,
    "train_targets":train_targets,
    "test_samples":test_samples,
    "test_targets":test_targets
}


# In[219]:

ML_inputs = ML_inputs_tuple(ML_inputs)


# In[220]:

estimator_name = "Log_C"
clf = run_ML(cf=None,ML_inputs=ML_inputs, estimator_name=estimator_name)


# In[233]:

print np.argsort(clf.best_estimator_.coef_[0])[::-1]
clf.best_estimator_.coef_[0][np.argsort(clf.best_estimator_.coef_[0])[::-1]]


# In[227]:

np.argsort(clf.best_estimator_.coef_[0])[::-1]


# In[232]:

clf.best_estimator_.intercept_


# In[231]:

clf.best_estimator_.coef_[0]


# In[235]:

np.savetxt("feature_weights.txt",clf.best_estimator_.coef_[0])


# In[183]:

with open("7mers.txt","r") as f:
    features = f.readlines()


# In[186]:

features = features[0].split("\t")


# In[234]:

features[644]


# In[236]:

with open("feature_weights.txt","r") as f:
    feature_weights = f.readlines()

