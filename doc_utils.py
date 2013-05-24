from collections import Counter
import os,sys
from os import path
from math import log,exp
import re
import types
import cPickle as marshal


class DocUtils(object):
    '''Container class for utility static methods'''
    #inparams
    #  featureFile: input file containing queries and url features
    #return value
    #  queries: map containing list of results for each query
    #  features: map containing features for each (query, url, <feature>) pair
    @staticmethod
    def extractFeatures(featureFile):
        f = open(featureFile, 'r')
        queries = {}
        features = {}
    
        for line in f:
          key = line.split(':', 1)[0].strip()
          value = line.split(':', 1)[-1].strip()
          if(key == 'query'):
            query = value
            queries[query] = []
            features[query] = {}
          elif(key == 'url'):
            url = value
            queries[query].append(url)
            features[query][url] = {}
          elif(key == 'title'):
            features[query][url][key] = value
          elif(key == 'header'):
            curHeader = features[query][url].setdefault(key, [])
            curHeader.append(value)
            features[query][url][key] = curHeader
          elif(key == 'body_hits'):
            if key not in features[query][url]:
              features[query][url][key] = {}
            temp = value.split(' ', 1)
            features[query][url][key][temp[0].strip()] \
                        = [int(i) for i in temp[1].strip().split()]
          elif(key == 'body_length' or key == 'pagerank'):
            features[query][url][key] = int(value)
          elif(key == 'anchor_text'):
            anchor_text = value
            if 'anchors' not in features[query][url]:
              features[query][url]['anchors'] = {}
          elif(key == 'stanford_anchor_count'):
            features[query][url]['anchors'][anchor_text] = int(value)
          
        f.close()
        return (queries, features)
    
    @staticmethod
    def compute_tf_vector(words,multiplier = 1):
        tf = {}
        for w in words:
            if w not in tf: tf[w] = 0.0
            tf[w] += 1
        if multiplier > 1:
            for w in tf: tf[w] *= multiplier
        return tf
    
    @staticmethod          
    def url_tf_vector(url):
        words = filter(lambda x: len(x) > 0,re.split('\W',url))
        return DocUtils.compute_tf_vector(words)

    @staticmethod
    def header_tf_vector(header):
        words = reduce(lambda x,h: x+h.strip().lower().split(),header,[])
        return DocUtils.compute_tf_vector(words)

    @staticmethod
    def body_tf_vector(body_hits):
        tf = {}
        for bh in body_hits:
            tf[bh] = len(body_hits[bh])     
        return tf

    @staticmethod
    def title_tf_vector(title): 
        words = title.lower().strip().split() # Can do stemming etc here
        return DocUtils.compute_tf_vector(words)

    @staticmethod
    def anchor_tf_vector(anchors):
        tf = {}
        for a in anchors:
            atf = a.term_counts
            for term in atf:
                if term not in tf: tf[term] = 0.0
                tf[term] += atf[term]
        return tf
    
class Page(object):
    
    fields = ['url','header','body','anchor','title']
    
    '''Represents a single web page, with all its fields. Contains TF vectors for the fields'''
    def __init__(self,page,page_fields):
        self.url = page
         
        self.body_length = page_fields.get('body_length',1.0)
        self.body_length = max(1000.0,self.body_length) #(500.0 if self.body_length == 0 else self.body_length)
        
        self.pagerank         = page_fields.get('pagerank',0)
        self.title            = page_fields.get('title',"")
        self.header           = page_fields.get('header',"")
        self.body_hits        = page_fields.get('body_hits',{})
        self.anchors          = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
        self.field_tf_vectors = self.compute_field_tf_vectors()
                        

    def compute_field_tf_vectors(self):
        tfs = {}
        tfs['url']      = DocUtils.url_tf_vector(self.url)
        tfs['header']   = DocUtils.header_tf_vector(self.header)
        tfs['body']     = DocUtils.body_tf_vector(self.body_hits)   
        tfs['title']    = DocUtils.title_tf_vector(self.title)
        tfs['anchor']   = DocUtils.anchor_tf_vector(self.anchors)
        
        return tfs
    
    def get_field_tf(self, field_name, query_terms):
        tf = []
        tfs = self.field_tf_vectors
        
        for term in query_terms:
            tf.append( 0 if term not in tfs[field_name] else tfs[field_name][term])
            
        return tf
        
    
    
class Anchor(object):
    '''Properties of a single anchor text chunk'''
    def __init__(self,anchor_text,anchor_count):
        self.text = anchor_text
        self.terms = self.text.lower().strip().split()
        self.count = anchor_count
        self.term_counts = DocUtils.compute_tf_vector(self.terms,self.count)    

class Query(object):
    '''A single query, with all the results associated with it'''

    def __init__(self,query,query_pages,corpus=None):  # query_pages : query -> urls
        self.query = query

        self.terms = self.query.lower().strip().split()
        self.query_tf_vector = DocUtils.compute_tf_vector(self.terms)
        
        self.pages = [Page(p,v) for p,v in query_pages.iteritems()]
        
