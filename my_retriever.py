import math
from collections import defaultdict


class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting

        # total number of documents in a collection
        self.collection_size = int
        # total number of terms in a document -- {docid: count}
        self.n_terms = defaultdict(int)
        # inverse document frequency for each term -- {term: idf_value}
        self.idf = dict()

        # document weight vector -- {docid: {term: weight}}
        self.document_weight_vector = defaultdict(dict)
        # length of the document weight vector -- {docid: length}
        self.len_document_weight_vector = defaultdict(float)

        ##############################################
        # Calculate the collection size
        all_documentIds = set()
        for term in self.index:
            for docid in self.index[term]:
                all_documentIds.add(docid)

        self.collection_size = len(all_documentIds)

        ##############################################

        # TF-IDF scheme
        if self.termWeighting == 'tfidf':
            # inverse document frequency for each term -- self.idf = {term: idf_value}
            # term frequency in a document -- self.index[term][docid]

            # Calculate Inverse Document Frequency
            for term in self.index:
                # total number of documents : self.collection_size
                # number of documents with term : len(self.index[term])
                self.idf[term] = math.log(self.collection_size/len(self.index[term]), 10)

            # Calculate TF-IDF
            for term in self.index:
                for docid in self.index[term]:
                    self.document_weight_vector[docid].update({term: self.index[term][docid]*self.idf[term]})

        # Term Frequency scheme
        elif self.termWeighting == 'tf':
            for term in self.index:
                for docid in self.index[term]:
                    # number of times term appears in a document : self.index[term][docid]
                    self.document_weight_vector[docid].update({term: self.index[term][docid]})

        # Binary scheme
        else:
            for term in self.index:
                for docid in self.index[term]:
                    # binary weights for index terms in a document
                    self.document_weight_vector[docid].update({term: 1})

        # 1. Calculate the length of the document weight vector
        for docid, weight in self.document_weight_vector.items():
            d = list(weight.values())
            length = math.sqrt(sum([a*a for a in d]))
            self.len_document_weight_vector[docid] = length

    ##############################################
    # Method performing retrieval for specified query

    def forQuery(self, query):

        scores = defaultdict(float)

        # 2. Calculate the length of a query vector
        query_vector = []
        # tf-idf
        if self.termWeighting == 'tfidf':
            for term in query:
                if term not in self.idf:
                    pass
                else:
                    query_vector.append(query[term]*self.idf[term])
            len_query = math.sqrt(sum([a*a for a in query_vector]))

        # term frequency
        elif self.termWeighting == 'tf':
            for term in query:
                query_vector.append(query[term])
            len_query = math.sqrt(sum([a*a for a in query_vector]))

        # binary
        else:
            len_query = math.sqrt(len(query))

        ##############################################

        # Calculate Cosine Similarity
        for docid in range(self.collection_size):
            q = 0  # query vector element
            d = 0  # document vector element
            numerator = 0  # dot product of q, d

            for term in query:
                # Calculate only the term in query which exists in the document
                if term in self.document_weight_vector[docid+1]:
                    # 3. numerator = dot product of query and document vector
                    if self.termWeighting == 'tfidf':
                        q = query[term] * self.idf[term]
                        d = self.index[term][docid+1]*self.idf[term]
                        numerator += q * d

                    elif self.termWeighting == 'tf':
                        q = query[term]
                        d = self.index[term][docid+1]
                        numerator += q * d
                    else:
                        q += 1
                        d += 1
                        numerator += q * d

            # Cosine Similarity
            denominator = len_query * self.len_document_weight_vector[docid+1]
            score = numerator/denominator

            scores[docid+1] = score

        # best_rank = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        best_rank_10 = sorted(scores, key=scores.get, reverse=True)[:10]

        return best_rank_10

