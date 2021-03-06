from collections import Counter
import os,sys
from os import path
from math import log,exp
import numpy as np
import re
import types
import cPickle as marshal
from sklearn import preprocessing
from itertools import combinations

class DocUtils(object):

    @staticmethod    
    def getRankedQueries(queries,index_map,y):
        rankedQueries = {}
        for query in queries:
          results = [url for url in index_map[query] ]
          rankedQueries[query] = sorted(results, key = lambda url: y[index_map[query][url]], reverse = True)
    
        return rankedQueries
        
    #inparams
    #  queries: contains ranked list of results for each query
    #  outputFile: output file name
    @staticmethod
    def printRankedResults(queries,outFileName):
        with open(outFileName,"wb") as outfile:
            for query in queries:
                print("query: " + query)
                print >> outfile, ("query: " + query)
                for res in queries[query]:
                    print("  url: " + res)                
                    print >> outfile, ("  url: " + res)  

    @staticmethod
    def extractXy_pairWise(query_url_file, query_url_relevance_file,corpus=None, extraFeaturesInfo=None):
        X_p = []
        y_p = []
        
        (q,features) = DocUtils.extractFeatures(query_url_file)
        queries      = [Query(query,features[query],corpus) for query in features]
        relevances   = DocUtils.extractRelevances(query_url_relevance_file)        
        
        query_indices = {}
        count = 0
        for query in queries:
            query_indices[query] = set()
            query_tf = np.array(query.query_tf_vector)
          
            for page in query.pages:
                xrow = DocUtils.compute_x_row(query.terms,query_tf,page)
                yrow = relevances[query.string][page.url]
                
                if extraFeaturesInfo:
                    xrow_extra_features = DocUtils.compute_x_row_extra_features(query.string, page, extraFeaturesInfo)
                    xrow                = np.append(xrow, xrow_extra_features)
                
                X_p.append(xrow)
                y_p.append(yrow)
                query_indices[query].add(count)
                count += 1
     
        
        X_p = preprocessing.scale(X_p)
        X = []
        y = []
        
        for query in queries:
            indices = query_indices[query]       
            # Now do the pairwise
            for i,j in combinations(indices,2):
                if y_p[i] != y_p[j]:
                    X.append(X_p[i] - X_p[j])
                    y.append(1 if y_p[i] > y_p[j] else -1)
        
        return (X,y)
  
    @staticmethod
    def extractX_pairWise(query_url_file,corpus=None,extraFeaturesInfo=None):
        X,queries,X_index_map = DocUtils.extractX_pointWise(query_url_file, corpus, extraFeaturesInfo)
        # Scale X
        X = preprocessing.scale(X)
        return (X,queries,X_index_map)

    @staticmethod
    def extractXy_pointWise(query_url_file, query_url_relevance_file, corpus=None, extraFeaturesInfo=None):
        X = []
        y = []
       
        (q,features) = DocUtils.extractFeatures(query_url_file)
        queries      = [Query(query,features[query],corpus) for query in features]
        relevances   = DocUtils.extractRelevances(query_url_relevance_file)        
        
        for query in queries:
          query_tf = np.array(query.query_tf_vector)
          
          for page in query.pages:
            x_row_features = DocUtils.compute_x_row(query.terms,query_tf,page)
            
            if extraFeaturesInfo:
                x_row_extra_features = DocUtils.compute_x_row_extra_features(query.string, page, extraFeaturesInfo)
                x_row_features       = np.append(x_row_features, x_row_extra_features)
            
            X.append(x_row_features)
            y.append(relevances[query.string][page.url])
            
        return (X,y)


    @staticmethod
    def extractX_pointWise(query_url_file, corpus=None, extraFeaturesInfo=None):
        X           = []
        queries     = []
        X_index_map = {}
        count       = 0
       
        (q,features) = DocUtils.extractFeatures(query_url_file)
        queryObjects = [Query(query,features[query],corpus) for query in features]
        
        for query in queryObjects:
          query_tf = np.array(query.query_tf_vector)
          queries.append(query.string)
          X_index_map[query.string] = {}
          
          for page in query.pages:
            x_row_features = DocUtils.compute_x_row(query.terms,query_tf,page)
            
            if extraFeaturesInfo:
                x_row_extra_features = DocUtils.compute_x_row_extra_features(query.string, page, extraFeaturesInfo)
                x_row_features       = np.append(x_row_features, x_row_extra_features)
                
            X.append(x_row_features)
            #print >> sys.stderr, "Test: ", x_row_features
            X_index_map[query.string][page.url] = count
            count = count + 1
            
        return (X,queries,X_index_map)

    @staticmethod
    def compute_x_row(query_terms, query_tf, page):
        url_tf    = np.array(page.get_field_tf('url',query_terms))
        title_tf  = np.array(page.get_field_tf('title',query_terms))
        header_tf = np.array(page.get_field_tf('header',query_terms))
        body_tf   = np.array(page.get_field_tf('body',query_terms))
        anchor_tf = np.array(page.get_field_tf('anchor',query_terms))
        
        x_row = np.array([np.dot(query_tf,url_tf),
                          np.dot(query_tf,title_tf),
                          np.dot(query_tf,header_tf),
                          np.dot(query_tf,body_tf),
                          np.dot(query_tf,anchor_tf)])
                    
        return x_row
    
    @staticmethod
    def compute_x_row_extra_features(query_string, page, extraFeaturesInfo):
        extra_features = np.array([])
        
        isPDF       = 1. if page.url.endswith(".pdf") else 0.
        pagerank    = float(page.pagerank)
        urltoks     = float(len(page.field_tf_vectors['url']))
        tittoks     = float(len(page.field_tf_vectors['title']))
        hasQ        = 1.0 if '?' in page.url else 0.
        
        urlranks = 0
        urltokens = filter(lambda x: len(x) > 0,re.split('\W',page.url))
        queries = query_string.strip().split()
        for q in queries:
            try:
                urlranks += (urltokens.index(q)+1)
            except:
                pass
        
        bm25f_score = float(extraFeaturesInfo.bm25f_scores[query_string][page.url])
        window_sizes = extraFeaturesInfo.window_sizes[query_string][page.url][3]

        extra_features = np.append(extra_features, isPDF)
        extra_features = np.append(extra_features, pagerank)
        extra_features = np.append(extra_features, window_sizes)
        extra_features = np.append(extra_features, bm25f_score)
        extra_features = np.append(extra_features, urltoks)
        extra_features = np.append(extra_features, tittoks)
        extra_features = np.append(extra_features, hasQ)
        #extra_features = np.append(extra_features, urlranks)
        
        return extra_features
    
        
    #inparams
    #  featureFile: input file containing relevances per query-url
    #return value
    #  relevances: map containing relevances for each (query, url) pair
    @staticmethod
    def extractRelevances(relevanceFile):
        f = open(relevanceFile, 'r')
        relevances = {}
    
        for line in f:
          key = line.split(':', 1)[0].strip()
          value = line.split(':', 1)[-1].strip()
          if(key == 'query'):
            query = value
            relevances[query] = {}
          elif(key == 'url'):
            url = value.split(' ', 1)[0].strip()
            relevance = value.split(' ', 1)[-1].strip()
            relevances[query][url] = float(relevance)
            
        return relevances

    #inparams
    #  featureFile: input file containing window sizes per query-url
    #return value
    #  relevances: map containing window sizes for each (query, url) pair
    @staticmethod
    def extractWindowSizes(windowSizesFile):
        f = open(windowSizesFile, 'r')
        windowSizes = {}
    
        for line in f:
          key = line.split(':', 1)[0].strip()
          value = line.split(':', 1)[-1].strip()
          if(key == 'query'):
            query = value
            windowSizes[query] = {}
          elif(key == 'url'):
            url = value.split(' ', 1)[0].strip()
            # [url_windowSize, title_windowSize, header_windowSize, body_windowSize, anchor_windowSize]
            windowSizeList = [int(i) for i in value.split(' ', 1)[-1].strip().split(" ")]
            windowSizes[query][url] = windowSizeList
            
        return windowSizes


    
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
            tf[bh] = float(len(body_hits[bh]))
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
    
    NORMALIZE=True
    @staticmethod
    def normalize(otf,length):
        if not DocUtils.NORMALIZE: return otf
        length=float(length)
        tf = {}
        for w in otf:
            tf[w] = otf[w]/length
        return tf
               
    LOGIFY=True
    @staticmethod
    def logify(otf):
        if not DocUtils.LOGIFY: return otf
        tf = {}
        for w in otf:
            tf[w] = (1 + log(otf[w])) if otf[w] > 0 else 0
        return tf
    
    IDFY=True
    @staticmethod
    def IDFy(otf,corpus = None):
        if not DocUtils.IDFY: return otf
        if corpus is None: return otf
        tf = {}
        for w in otf:
            tf[w] = otf[w]*corpus.get_IDF(w)
        return tf

       
class Page(object):
    
    fields = ['url','header','body','anchor','title']
    
    '''Represents a single web page, with all its fields. Contains TF vectors for the fields'''
    def __init__(self,page,page_fields,corpus = None):
        self.corpus = corpus                
        self.url = page        
        self.body_length = page_fields.get('body_length',1.0)
        self.body_length = max(1750.0,self.body_length) 
        self.pagerank         = page_fields.get('pagerank',0)
        self.title            = page_fields.get('title',"")
        self.header           = page_fields.get('header',"")
        self.body_hits        = page_fields.get('body_hits',{})
        self.anchors          = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
        self.field_tf_vectors = self.compute_field_tf_vectors()

    def compute_field_tf_vectors(self):
        tfs = {}
        tfs['url']      = DocUtils.logify(DocUtils.url_tf_vector(self.url))
        tfs['header']   = DocUtils.logify(DocUtils.header_tf_vector(self.header))
        tfs['body']     = DocUtils.logify(DocUtils.normalize(DocUtils.body_tf_vector(self.body_hits),self.body_length))  
        tfs['title']    = DocUtils.logify(DocUtils.title_tf_vector(self.title))
        tfs['anchor']   = DocUtils.logify(DocUtils.anchor_tf_vector(self.anchors))
        
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
        self.string = query
        raw_terms = self.string.lower().strip().split() # it may include repeated terms
        
        tf_vector_dict = DocUtils.IDFy(DocUtils.compute_tf_vector(raw_terms),corpus)
        
        self.terms           = [] # it does not include repeated terms
        self.query_tf_vector = [] # raw frequency of each term in the query string
        
        for k,v in tf_vector_dict.iteritems():
            self.terms.append(k)
            self.query_tf_vector.append(v)
        
        self.pages = [Page(p,v,corpus) for p,v in query_pages.iteritems()]


class ExtraFeaturesInfo(object):
    '''ExtraFeatures to be used in the classification i.e. bm25f scores and field window_sizes'''
    
    def load(self, bm25fFile, windowSizesFile):
        extractScores     = DocUtils.extractRelevances  # bm25f scores file has same format as relevances file
        self.bm25f_scores = extractScores(bm25fFile)
        self.window_sizes = DocUtils.extractWindowSizes(windowSizesFile)
        

class CorpusInfo(object):
    '''Represents a corpus, which can be queried for IDF of a term'''
    def __init__(self,corpus_root_dir=None): # for Laplace smoothing
        self.corpus_dir = corpus_root_dir
        self.total_file_count = 1.0
        self.df_counter = Counter()   # term -> doc_freq
        
    def compute_doc_freqs(self):
        root = self.corpus_dir
        for d in sorted(os.listdir(root)):
          print >> sys.stderr, 'processing dir: ' + d
          dir_name = os.path.join(root, d) 
          term_doc_list = []
          
          for f in sorted(os.listdir(dir_name)):
            self.total_file_count += 1
            
            # Add 'dir/filename' to doc id dictionary
            file_name = os.path.join(d, f)

            fullpath = os.path.join(dir_name, f)
            
            with open(fullpath, 'r') as infile:
                lines = [line for line in infile.readlines()]
                tokens = set(reduce(lambda x,line: x+line.strip().split(),lines,[]))   
                for token in tokens: self.df_counter[token] += 1
        marshal.dump((self.total_file_count,self.df_counter),open("IDF.dat","wb"))
    
    def load_doc_freqs(self):
        if path.isfile("IDF.dat"):
            print >> sys.stderr, "Loading IDF from file"
            self.total_file_count,self.df_counter = marshal.load(open("IDF.dat"))
            print >> sys.stderr, "Size of corpus",len(self.df_counter)
        else:
            print >> sys.stderr, "Computing IDF"
            self.compute_doc_freqs()
        
    def get_IDF(self,term):
        idf = log(self.total_file_count)-log(self.df_counter[term]+1.0)
        #print >> sys.stderr, "IDF of term:",term,"is",idf
        return idf # for Laplace smoothing
