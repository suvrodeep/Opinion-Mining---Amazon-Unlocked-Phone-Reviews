# Databricks notebook source
from collections import defaultdict
from pyspark import SparkContext
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import SQLContext
import re

sc = SparkContext()

# COMMAND ----------

##num_of_stop_words = 2     # Number of most common words to remove, trying to eliminate stop words
num_topics = 10	            # Number of topics we are looking for
num_words_per_topic = 10    # Number of words to display for each topic
max_iterations = 50         # Max number of times to iterate before finishing

##Function
def topic_models(x , y , z):

  data_reviews = x

  stop = ["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]

  tokens = data_reviews.map( lambda document: document.strip().lower()).map( lambda document: re.split("[\s;,#]", document)).map( lambda word: [x for x in word if x.isalpha()]).map( lambda word: [x for x in word if len(x) > 2]).map(lambda x: [item for item in x if item not in stop])

  termCounts = tokens.flatMap(lambda document: document).map(lambda word: (word, 1)).reduceByKey( lambda x,y: x + y).map(lambda tuple: (tuple[1], tuple[0])).sortByKey(False)

  vocabulary = termCounts.map(lambda x: x[1]).zipWithIndex().collectAsMap()

  ## Internal FUNCTION
  def document_vector(document):
      id = document[1]
      counts = defaultdict(int)
      for token in document[0]:
          if token in vocabulary:
              token_id = vocabulary[token]
              counts[token_id] += 1
      counts = sorted(counts.items())
      keys = [x[0] for x in counts]
      values = [x[1] for x in counts]
      return (id, Vectors.sparse(len(vocabulary), keys, values))

  documents = tokens.zipWithIndex().map(document_vector).map(list)

  inv_voc = {value: key for (key, value) in vocabulary.items()}

  ## Running LDA
  lda_model = LDA.train(documents, k=num_topics, maxIterations=max_iterations)

  topic_indices = lda_model.describeTopics(maxTermsPerTopic=num_words_per_topic)

  ##Writing to file
  with open("/Users/prasoon/Documents/OneDrive/Spring_Semester_2016/BigData/Big_data_Project/output_complete_file.txt", 'a') as f:
      lda_model = LDA.train(documents, k=num_topics, maxIterations=max_iterations)

      topic_indices = lda_model.describeTopics(maxTermsPerTopic=num_words_per_topic)

      # Print topics, showing the top-weighted 10 terms for each topic
      for i in range(len(topic_indices)):
        ##f.write("Topic #{0}\n".format(i + 1))
        for j in range(len(topic_indices[i][0])):
          f.write("{0},{1},Topic_{2},{3},{4}\n".format(y , z, (i+1), inv_voc[topic_indices[i][0][j]].encode('utf-8'), topic_indices[i][1][j]))
            

data = sc.textFile('/Users/prasoon/Documents/OneDrive/Spring_Semester_2016/BigData/Big_data_Project/Amazon_Unlocked_Mobile_suvro.csv').map(lambda x : x.split(',' ,  4))


for brands in ["HTC" , "Samsung" , "Apple", "Microsoft" , "BlackBerry" , "Nokia" , "Huawei" , "Motorola" , "BLU" , "LG" , "Sony" ,  "Posh Mobile"]:
  for i in ["1","2","3","4","5"]:
    data_reviews = data.filter(lambda line : (line[0] == brands and line[2] == i )).map(lambda x : x[4])
    topic_models(data_reviews , brands , i)
## 

##["HTC" , "Samsung" , "Apple", "Microsoft" , "BlackBerry" , "Nokia" , "Huawei" , "Motorola" , "BLU" , "LG" , "Sony" ,  "Posh Mobile"]
