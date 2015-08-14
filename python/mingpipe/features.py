class FeatureExtractor:
    def __init__(self, mode=17, align_mode='align'):
        self.mode = int(mode)
        self.align_mode = align_mode
        #TODO: Abstract this.
        ''' a similarity array for feature(n-gram overlap binary feature) extraction '''
        self.simArray = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.separator = '--'
        self.featureDict = {}
        self.ngrm_idfDict = {}
        self.lastnameSet = set()
        self.sim_handler = None
        self.ngram = 3
        self.longest_diff = 5


    def saveFeatureDict(self, featureDictFile):
        import pickle
        pickle.dump(self.featureDict, open(featureDictFile, 'wb'))


    def loadFeatureDict(self, featureDictFile):
        import pickle
        self.featureDict = pickle.load( open(featureDictFile, 'rb') )


    def loadLastnameSet(self, dict_file):
        import codecs as cs
        with cs.open(dict_file, 'r', encoding='utf-8') as df:
            for line in df:
                elems = line.rstrip().split()
                for nm in elems:
                    self.lastnameSet.add(nm)



    def constructPairFeatureDict(self, str1, str2):
        str1_ngramSet = self.__extractNgramUnion(str1)
        str2_ngramSet = self.__extractNgramUnion(str2)
        for ngm in str1_ngramSet:
            for ngrm in str2_ngramSet:
                pair = ngm + self.separator + ngrm 
                if pair not in self.featureDict:
                    self.featureDict[pair] = [len(self.featureDict), 1]
                else:
                    self.featureDict[pair][1] += 1
        str1_uniqueSet = str1_ngramSet - str2_ngramSet
        for ngm in str1_uniqueSet:
            pair = ngm + self.separator
            if pair not in self.featureDict:
                self.featureDict[pair] = [len(self.featureDict), 1]
            else:
                self.featureDict[pair][1] += 1
        str2_uniqueSet = str2_ngramSet - str1_ngramSet
        for ngm in str2_uniqueSet:
            pair = self.separator + ngm
            if pair not in self.featureDict:
                self.featureDict[pair] = [len(self.featureDict), 1]
            else:
                self.featureDict[pair][1] += 1



    ''' Helper function to construct str_pair dict'''
    def __extractNgramUnion(self, targetStr):
        ngram_set = set()
        for i, char in enumerate(targetStr):
            for n in xrange(self.ngram):
                if i+n < len(targetStr):
                    ngram_set.add(' '.join(targetStr[i:i+n+1]))
        return ngram_set

    # New implementation of alignment: extract ngrams and align.
    def constructAlignedFeatureDict(self, str1, str2, sim_handler):
        str1_ngram = list(self.__extractNgramUnion(str1))
        str2_ngram = list(self.__extractNgramUnion(str2))
        from .utils import alignment
        self.sim_handler = sim_handler
        align_idx = alignment(str1_ngram, str2_ngram, self.sim_handler)
        idx1_set = set()
        idx2_set = set()
        for elem in align_idx:
            idx1, idx2 = elem
            idx1_set.add(idx1)
            idx2_set.add(idx2)
            pair = str1_ngram[idx1] + self.separator + str2_ngram[idx2] 
            if pair not in self.featureDict:
                self.featureDict[pair] = [len(self.featureDict), 1]
            else:
                self.featureDict[pair][1] += 1
        if len(str1_ngram) == len(str2_ngram):
            return
        elif len(str1_ngram) > len(str2_ngram):
            unique_set = set(range(len(str1_ngram))) - idx1_set
            for el in unique_set:
                pair = str1_ngram[el] + self.separator
                if pair not in self.featureDict:
                    self.featureDict[pair] = [len(self.featureDict), 1]
                else:
                    self.featureDict[pair][1] += 1
                #print 'str1 have extra elem:', pair 
        else:
            unique_set = set(range(len(str2_ngram))) - idx2_set
            for el in unique_set:
                pair = self.separator + str2_ngram[el]
                if pair not in self.featureDict:
                    self.featureDict[pair] = [len(self.featureDict), 1]
                else:
                    self.featureDict[pair][1] += 1


    def construct_idfDict(self, str1, str2):
        str1_ngramSet = self.__extractNgramUnion(str1)
        str2_ngramSet = self.__extractNgramUnion(str2)
        for ngm in str1_ngramSet:
            if ngm not in self.ngrm_idfDict:
                self.ngrm_idfDict[ngm] = 1 
            #else:
            self.ngrm_idfDict[ngm] += 1
        for ngm in str2_ngramSet:
            if ngm not in self.ngrm_idfDict:
                self.ngrm_idfDict[ngm] = 1
            #else:
            self.ngrm_idfDict[ngm] += 1


    def filter_feature(self, threshold):
        for key, val in self.featureDict.items():
            if val[1] < threshold:
                del self.featureDict[key]
        count = 0
        for key in self.featureDict:
            self.featureDict[key][0] = count
            count += 1


    def extract(self, name1, name2):
        #raw_name1, string1, pinyin1, pronun1 = name1
        #raw_name2, string2, pinyin2, pronun2 = name2
        #print "extracting pair:", name1[0], name2[0]
        if self.mode >= 18 :
            raise ValueError('Wrong mode for feature extraction! Currently available modes are n-gram overlap mode and skip-gram overlap mode!')
        charset_idx = self.mode / 6 + 1 #7 - 3
        str1 = name1[charset_idx]
        str2 = name2[charset_idx]
        leave_idx = self.mode % 6 #7
        features = {}
        begin = 1
        if leave_idx != 0:
            self.__appendNgramOverlapFeat_bin(features, name1[0], name2[0], begin)
            begin += self.ngram*len(self.simArray)+1
            self.__appendNgramOverlapFeat_bin(features, str1, str2, begin)
            begin += self.ngram*len(self.simArray)+1
        if leave_idx != 1:
            self.__appendLastnameMatch(features, name1[0], name2[0], begin)
            begin += 2
        if leave_idx != 2:
            self.__appendLengthDiff(features, name1[0], name2[0], begin)
            begin += self.longest_diff + 1
        if leave_idx != 3:
            self.__appendDistanceMeasure(features, str1, str2, begin, 1)
            begin += len(self.simArray) #1
        #print 'features before align:', features
        if leave_idx != 4:
            if self.align_mode == 'align':
                self.__appendAlignPairFeat_bin(features, name1[0], name2[0], begin)
            elif self.align_mode == 'all':
                self.__appendNgramPairFeat_bin(features, name1[0], name2[0], begin)
            else:
                raise ValueError('Wrong mode for alignment feature extraction! Currently available modes are alignmode and all mode!')
            #rint raw_name1, raw_name2, pinyin1, pinyin2
        if leave_idx != 5:
            if self.align_mode == 'align': 
                self.__appendAlignPairFeat_bin(features, str1, str2, begin)
            elif self.align_mode == 'all':
                self.__appendNgramPairFeat_bin(features, str1, str2, begin)

        #print 'final features:', features
        return features


    ''' extract ngrams from a given string'''
    def __extractNgrams(self, targetStr):
        ngram_set=[set() for i in range(self.ngram)]
        for i, char in enumerate(targetStr):
            for n in xrange(self.ngram):
                if i+n < len(targetStr):
                    ngram_set[n].add(' '.join(targetStr[i:i+n+1]))
        return ngram_set


    ''' generate ngram overlap features = jaccard measure''' 
    def __appendNgramOverlapRate(self, featureMap, str1, str2, begin):
        str1_ngramSet = self.__extractNgrams(str1)
        str2_ngramSet = self.__extractNgrams(str2)
        #print "string1 ngram set = ", str1_ngramSet 
        #print "string2 ngram set = ", str2_ngramSet
        non_overlap = True
        for i, (elem1, elem2) in enumerate(zip(str1_ngramSet, str2_ngramSet)):
            overlap = elem1 & elem2
            if len(overlap) == 0:
                continue
            non_overlap = False
            union = elem1 | elem2
            #print elem1, elem2, overlap, union
            tempSim = float(len(overlap)) / len(union)
            #print str(i+1),'gram overlap fire!!!', tempSim
            #featureArray.append(':'.join( (str(begin+2*len(self.simArray)+i), '1')))
            featureMap[begin+i] = tempSim
        if non_overlap:
            #print 'no ngram overlap!!', str1, str2
            featureMap[begin+self.ngram] = 1



    ''' generate ngram overlap binary features''' 
    def __appendNgramOverlapFeat_bin(self, featureMap, str1, str2, begin):
        str1_ngramSet = self.__extractNgrams(str1)
        str2_ngramSet = self.__extractNgrams(str2)
        #print "string1 ngram set = ", str1_ngramSet 
        #print "string2 ngram set = ", str2_ngramSet
        non_overlap = True
        for i, (elem1, elem2) in enumerate(zip(str1_ngramSet, str2_ngramSet)):
            overlap = elem1 & elem2
            if len(overlap) == 0:
                continue
            non_overlap = False
            union = elem1 | elem2
            #print elem1, elem2, overlap, union
            tempSim = float(len(overlap)) / len(union)
            #print str(i+1),'gram overlap fire!!!', tempSim
            self.binarize(begin+i*len(self.simArray), tempSim, featureMap)
        if non_overlap:
            #print 'no ngram overlap!!', str1, str2
            featureMap[begin+self.ngram*len(self.simArray)] = 1



    def binarize(self, begin, tempSim, featureMap):
        for i in range(len(self.simArray)-1,-1,-1):
            if tempSim > self.simArray[i]:
                featureMap[begin+i] = 1
                featureMap[begin+i+1] = 1
                return	



    ''' generate ngram pair features''' 
    def __appendNgramPairFeat_bin(self, featureMap, str1, str2, begin):
        import sys
        if len(self.featureDict) == 0:
            sys.exit('feature dictionary is empty!!')
        str1Ngram = self.__extractNgramUnion(str1) 
        str2Ngram = self.__extractNgramUnion(str2)

        #print 'str1 ngram union:', str1Ngram
        #print 'str2 ngram union:', str2Ngram
        for ngm in str1Ngram:
            for ngrm in str2Ngram:
                pair = ngm + self.separator + ngrm 
                if pair not in self.featureDict:
                    continue
                #print pair
                featureMap[begin + self.featureDict[pair][0]] = 1
        str1_uniqueSet = str1Ngram - str2Ngram
        for ngm in str1_uniqueSet:
            pair = ngm + self.separator
            if pair not in self.featureDict:
                continue
            else:
                featureMap[begin + self.featureDict[pair][0]] = 1
        str2_uniqueSet = str2Ngram - str1Ngram
        for ngm in str2_uniqueSet:
            pair = self.separator + ngm
            if pair not in self.featureDict:
                continue
            else:
                featureMap[begin + self.featureDict[pair][0]] = 1



    ''' generate alignment pair features''' 
    def __appendAlignPairFeat_bin(self, featureMap, str1, str2, begin):
        if len(self.featureDict) == 0:
            import sys
            sys.exit('feature dictionary is empty!!')
        str1_ngram = list(self.__extractNgramUnion(str1))
        str2_ngram = list(self.__extractNgramUnion(str2))
        from .utils import alignment
        align_idx = alignment(str1_ngram, str2_ngram, self.sim_handler)
        idx1_set = set()
        idx2_set = set()
        for elem in align_idx:
            idx1, idx2 = elem
            idx1_set.add(idx1)
            idx2_set.add(idx2)
            pair = str1_ngram[idx1] + self.separator + str2_ngram[idx2] 
            if pair not in self.featureDict:
                continue
                #print pair
            featureMap[begin + self.featureDict[pair][0]] = 1
        if len(str1_ngram) == len(str2_ngram):
            return
        elif len(str1_ngram) > len(str2_ngram):
            unique_set = set(range(len(str1_ngram))) - idx1_set
            for el in unique_set:
                pair = str1_ngram[el] + self.separator
                if pair not in self.featureDict:
                    continue
                else:
                    featureMap[begin + self.featureDict[pair][0]] = 1
        else:
            unique_set = set(range(len(str2_ngram))) - idx2_set
            for el in unique_set:
                pair = self.separator + str2_ngram[el]
                if pair not in self.featureDict:
                    continue
                else:
                    featureMap[begin + self.featureDict[pair][0]] = 1


    '''compute length difference feature'''
    def __appendLengthDiff(self, featureMap, str1, str2, begin):
        len_diff = abs(len(str1) - len(str2))
        if len_diff >= self.longest_diff:
            featureMap[begin+self.longest_diff] = 1
        else:
            featureMap[begin+len_diff] = 1
        #print 'in length different function:', len_diff


    ''' indicator feature on whether the 1st char is matched of the 2 string'''
    def __appendLastnameMatch(self, featureMap, str1, str2, begin):
        from .utils import get_resource
        if len(self.lastnameSet) == 0:
            self.loadLastnameSet(get_resource('Chinese_lastname'))
        if str1[0] == str2[0]:
            featureMap[begin] = 1
        if (str1[0] == str2[0] and str1[0] in self.lastnameSet) or (str1[0:2] == str2[0:2] and str1[0:2] in self.lastnameSet):
            featureMap[begin+1] = 1
        #print 'in last name match function:', featureMap.get(begin, 0)


    ''' Append similarity score/decision to the feature vector.'''
    def __appendDistanceMeasure(self, featureMap, str1, str2, begin, mode):
        from .similarity import levenshtein_distance, jaro_winkler, longest_common_substring
        if mode == 1:
            tempSim = levenshtein_distance(str1, str2)
        elif mode == 2:
            tempSim = jaro_winkler(str1, str2)
        elif mode == 3:
            tempSim = longest_common_substring(str1, str2)
        else:
            sys.stderr.exit('Wrong mode in distance measure!')
        self.binarize(begin, tempSim, featureMap)
        #print 'distance measure', mode, ':', featureMap


    def __appendTfidfMeasure(self, featureMap, str1, str2, begin):
        if len(self.ngrm_idfDict) == 0:
            import sys
            sys.exit('ngrm idf dictionary is empty!!')
        str1_map = self.__extractNgramMap(str1)
        str2_map = self.__extractNgramMap(str2)
        numerator = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for key, val in str1_map.items():
            if key in str2_map:
                tfidf1 = float(val)/self.ngrm_idfDict.get(key, 1)
                tfidf2 = float(str2_map[key]) / self.ngrm_idfDict.get(key, 1)
                numerator += tfidf1 * tfidf2
                norm1 += pow(tfidf1, 2)
                norm2 += pow(tfidf2, 2)
                #print 'key:', key, 'numerator:', numerator, 'norm1:i', norm1, 'norm2', norm2
        from math import sqrt
        if numerator == 0:
            return	
        cos_sim = numerator / sqrt(norm1*norm2)
        #featureMap[begin] = cos_sim
        self.binarize(begin, cos_sim, featureMap)



    ''' helper function to extract ngram maps map ngram to the occurrence.'''
    def __extractNgramMap(self, targetStr):
        ngram_map = {} 
        for i, char in enumerate(targetStr):
            for n in xrange(self.ngram):
                if i+n < len(targetStr):
                    key = ' '.join(targetStr[i:i+n+1])
                    if key not in ngram_map:
                        ngram_map[key] = 0
                    ngram_map[key] += 1
        return ngram_map
