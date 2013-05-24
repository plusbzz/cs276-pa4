#!/usr/bin/env python

### Module imports ###
import sys
import math
import re
import numpy as np
from sklearn import linear_model, svm
from sklearn import preprocessing
from doc_utils import DocUtils, Query

###############################
##### Point-wise approach #####
###############################
def pointwise_train_features(train_data_file, train_rel_file):
  X,y = DocUtils.extractXy_pointWise(train_data_file, train_rel_file)
  return (X, y)
 
def pointwise_test_features(test_data_file):
  X,queries,index_map = DocUtils.extractX_pointWise(test_data_file)
  return (X, queries, index_map)
 
def pointwise_learning(X, y):
  model = linear_model.LinearRegression()
  model.fit(X,y)
  return model

def pointwise_testing(X, model):
  y = model.predict(X) 
  return y

##############################
##### Pair-wise approach #####
##############################
def pairwise_train_features(train_data_file, train_rel_file):
  X,y = DocUtils.extractXy_pairWise(train_data_file, train_rel_file)
  return (X, y)
 
def pairwise_test_features(test_data_file):
  X,queries,index_map = DocUtils.extractX_pairWise(test_data_file)
  return (X, queries, index_map)

def pairwise_learning(X, y):
  model = svm.SVC(kernel='linear', C=1.0)
  model.fit(X,y)
  return model

def pairwise_testing(X, model):
  weights = model.coef_/np.linalg.norm(model.coef_)
  weights = preprocessing.scale(weights.T)
  y = np.dot(X,weights).T
  return y[0]

####################
##### Training #####
####################
def train(train_data_file, train_rel_file, task):
  sys.stderr.write('\n## Training with feature_file = %s, rel_file = %s ... \n' % (train_data_file, train_rel_file))
  
  if task == 1:
    # Step (1): construct your feature and label arrays here
    (X, y) = pointwise_train_features(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pointwise_learning(X, y)
  elif task == 2:
    # Step (1): construct your feature and label arrays here
    (X, y) = pairwise_train_features(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pairwise_learning(X, y)
  elif task == 3: 
    # Add more features
    print >> sys.stderr, "Task 3\n"

  elif task == 4: 
    # Extra credit 
    print >> sys.stderr, "Extra Credit\n"

  else: 
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    model = linear_model.LinearRegression()
    model.fit(X, y)
  
  # some debug output
  weights = model.coef_
  print >> sys.stderr, "Weights:", str(weights)

  return model 

###################
##### Testing #####
###################
def test(test_data_file, model, task):
  sys.stderr.write('\n## Testing with feature_file = %s ... \n' % (test_data_file))

  if task == 1:
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pointwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pointwise_testing(X, model)
    
  elif task == 2:
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pairwise_testing(X, model)
  elif task == 3: 
    # Add more features
    print >> sys.stderr, "Task 3\n"

  elif task == 4: 
    # Extra credit 
    print >> sys.stderr, "Extra credit\n"

  else:
    queries = ['query1', 'query2']
    index_map = {'query1' : {'url1':0}, 'query2': {'url2':1}}
    X = [[0.5, 0.5], [1.5, 1.5]]  
    y = model.predict(X)
  
  # some debug output
  for query in queries:
    print "query: "+query
    results = sorted([(url,y[index_map[query][url]]) for url in index_map[query]],
                        key=lambda tup: tup[1], reverse=True)
    for url,value in results:
      print "  url: "+url
      
      print >> sys.stderr, "Query:", query, ", url:", url, ", value:", value

  # Step (3): output your ranking result to stdout in the format that will be scored by the ndcg.py code

if __name__ == '__main__':
  sys.stderr.write('# Input arguments: %s\n' % str(sys.argv))
  
  if len(sys.argv) != 5:
    print >> sys.stderr, "Usage:", sys.argv[0], "train_data_file train_rel_file test_data_file task"
    sys.exit(1)
  
  train_data_file = sys.argv[1]
  train_rel_file = sys.argv[2]
  test_data_file = sys.argv[3]
  task = int(sys.argv[4])
  print >> sys.stderr, "### Running task", task, "..."
  
  model = train(train_data_file, train_rel_file, task)
  test(test_data_file, model, task)
