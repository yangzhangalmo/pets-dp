import os

import multiprocessing as mp

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from gensim.models import word2vec




def walk_core(model_name, G, start_node, walklen=80, walktimes=10):
    ''' random walks from start_node in the social network
    Args:
        model_name: pets-dp-2vec
        G: social network, networkx object
        start_node: starting user in a random walk
        walklen: walk length
        walktimes: times start from each node
    Returns:
    '''
    
    for i in range(walktimes):
        temp_walk = np.zeros((1, walklen))# initialize random walk
        temp_walk[:, 0] = start_node
        curr_node = start_node
        for j in range(walklen-1):
            temp_val = list(G.neighbors(curr_node))
            next_node = np.random.choice(temp_val, 1)[0]
            curr_node = next_node
            temp_walk[:, j+1] = next_node
        pd.DataFrame(temp_walk).to_csv('result/'+model_name+'.walk',\
                                       header=None, mode='a', index=False)

def walk(model_name, G, walklen=80, walktimes=10):
    ''' random walks in the social network
    Args:
        model_name: pets-dp-2vec
        G: social network, networkx object
        walklen: walk length
        walktimes: times start from each node
    Returns:
    '''

    nodelist = G.nodes.keys()

    print len(nodelist)

    for start_node in nodelist:
        walk_core(model_name, G, start_node, walklen, walktimes)

    print 'finish walking'

def learn_emb(model_name, walklen=80, walktimes=10, numfeatures=128):
    ''' learning word2vec vectors
    Args:
        model_name: pets-dp-2vec
        walk_len: walk length
        walktimes: times start from each node
        numfeatures: dimension of features
    Returns:
    '''

    print 'walk_len', walklen, 'walk_times', walktimes, 'num_features', numfeatures

    min_word_count = 5
    num_workers = mp.cpu_count()
    context = 10
    downsampling = 1e-3

    walk = pd.read_csv('result/'+model_name+'.walk',\
                       header=None, error_bad_lines=False)
    print 'walk_shape', walk.shape
    print walk.head(2)
    
    walk = walk.loc[np.random.permutation(len(walk))].reset_index(drop=True)

    walk = walk.loc[:,:walklen-1]
    walk = walk.groupby(0).head(walktimes).reset_index(drop=True)
    walk = walk.applymap(str) # gensim only accept list of strings
    print  walk.shape
    walk = walk.values.tolist()

    #skip-gram
    emb = word2vec.Word2Vec(walk,\
                            sg=1,\
                            workers=num_workers,\
                            size=numfeatures,\
                            min_count=min_word_count,\
                            window=context,\
                            sample=downsampling)

    print 'training done'
    emb.wv.save_word2vec_format('result/'+model_name+'.emb')

def build_pairs(user_list, friends):
    ''' construct friends and stranger pairs
    
    Args:
        user_list: a list of users
        friends: friends pairs, symmetric
    Returns:
        pairs: balanced friends (1) and stranger pairs (0)
    '''
    
    friends_pairs = friends.loc[friends.u1<friends.u2].reset_index(drop=True)

    strangers_pairs = pd.DataFrame(np.random.choice(user_list, 3*friends_pairs.shape[0]),\
                                   columns=['u1'])
    strangers_pairs['u2'] = np.random.choice(user_list, 3*friends_pairs.shape[0])

    strangers_pairs = strangers_pairs.loc[strangers_pairs.u1!=strangers_pairs.u2]
    strangers_pairs = strangers_pairs.loc[strangers_pairs.u1<strangers_pairs.u2]
    strangers_pairs = strangers_pairs.drop_duplicates().reset_index(drop=True)
    # delete friends inside
    strangers_pairs = strangers_pairs.loc[~strangers_pairs.set_index(list(strangers_pairs.columns)).index.\
                                          isin(friends_pairs.set_index(list(friends_pairs.columns)).index)]
    strangers_pairs = strangers_pairs.loc[np.random.permutation(strangers_pairs.index)].reset_index(drop=True)
    
    strangers_pairs = strangers_pairs.loc[0:1*friends_pairs.shape[0]-1, :]# down sampling

    friends_pairs['label'] = 1
    strangers_pairs['label'] = 0

    pairs = pd.concat([friends_pairs, strangers_pairs], ignore_index=True)
    
    return pairs

def build_feature(model_name):
    ''' construct features
    
    Args:
        model_name: pets-dp-2vec
    Returns:
    '''
    friends = pd.read_csv('dataset/snapfacebook.csv', names=['u1', 'u2'])
    user_list = friends.u1.unique()
    
    emb = pd.read_csv('result/'+model_name+'.emb', skiprows=1, header=None, sep=' ')

    pairs = build_pairs(user_list, friends)
    print pairs.shape
    
    for i in range(len(pairs)):
        u1 = pairs.loc[i, 'u1']
        u2 = pairs.loc[i, 'u2']
        label = pairs.loc[i, 'label']

        u1_emb = emb.loc[emb[0]==u1, range(1, 129)]
        u2_emb = emb.loc[emb[0]==u2, range(1, 129)]
        
        feature = (u1_emb.values*u2_emb.values)[0].tolist()
        feature.extend([label, u1, u2])
        pd.DataFrame([feature]).to_csv('result/'+model_name+'.feature', header=None, index=False, mode='a')
        
def friends_predict(model_name):
    ''' friendship prediction
    
    Args:
        model_name: pets-dp-2vec
    Returns:
    '''
    feature = pd.read_csv('result/'+model_name+'.feature', header=None)
    
    train_X, test_X, train_y, test_y = train_test_split(feature[range(128)], feature[128], test_size=0.3)
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=mp.cpu_count())
    clf.fit(train_X, train_y)
    proba_y = clf.predict_proba(test_X)[:, 1]
    predict_y = clf.predict(test_X)[:, 1]

    print 'AUC', roc_auc_score(test_y, proba_y)
    print 'precision', precision_score(test_y, predict_y)
    print 'recall', recall_score(test_y, predict_y)
    
if not os.path.exists('result/'):
    os.mkdir('result/')

model_name = 'pets-dp-2vec'

G = nx.read_edgelist('dataset/snapfacebook.csv', delimiter=',', nodetype=int)
walk(model_name, G, walklen=80, walktimes=10)

learn_emb(model_name)

build_feature(model_name)
friends_predict(model_name)