#!/usr/bin/python
# coding=utf-8

# A convenience method for accessing the matcher
matcher = None
def score(name1, name2, model_file=None, feature_file=None):
    global matcher
    if matcher is None:
        matcher = get_chinese_name_matcher(model_file=model_file, feature_file=feature_file)
        
    return matcher.predict(name1, name2)

def reset_model():
    global matcher
    matcher = None

def get_chinese_name_matcher(model_file=None, feature_file=None):
    from .features import FeatureExtractor
    from .utils import get_resource
    from .pinyin.langconv import Converter
    from .pinyin.ch2pinyin import Char2Pinyin
    from .pinyin.ch2pronun import Char2Pronun
    
    if not model_file:
        model_file = get_resource('chinese.model')
    if not feature_file:
        feature_file = get_resource('chinese.features')
        
    extractor = FeatureExtractor()
    default_matcher = NameMatcher(model_filename=model_file, threshold=0.45, \
                                    character_converter=Converter('zh-hans'), \
                                    pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), \
                                    pronun_converter=Char2Pronun(get_resource('mcpdict.csv')), \
                                    feature_extractor=extractor)
    default_matcher.load_feature_extractor(feature_file)

    return default_matcher


class NameMatcher:
    def __init__(self, model_filename=None, threshold=0.5, character_converter=None, pinyin_converter=None, pronun_converter=None, feature_extractor=None):
        # Load the model
        if model_filename:
            import pickle
            with open(model_filename, 'rb') as file:
                self.model = pickle.load(file)
                self.vectorizer = pickle.load(file)
        else:
            self.model = None

        ''' threshold that tuned by pre-training.'''
        self.threshold = threshold

        #TODO: Fix this.
        ''' default threshold for levenshtein edit distance'''
        #self.thre_levenshtein = 0.8 
        ''' default threshold for jaro-winkler distance'''
        #self.thre_jw = 0.5

        #TODO: Add handling in case this is None
        ''' a traditional Chinese to simplified Chinese character converter. '''
        self.character_converter = character_converter
        ''' a simplified Chinese character to Pinyin converter.'''

        #TODO: Figure out how to abstract this.
        #from utils import get_resource
        #from pinyin.ch2pinyin import Char2Pinyin
        self.pinyinConv = pinyin_converter #Char2Pinyin(get_resource('char2pinyin.new'))
        self.pronunConv = pronun_converter

        self.extractor = feature_extractor

    def set_threshold(self, thre):
        self.threshold = thre

    def train(self, data, options, cantonese=False):
        new_data = self._convert_data(data, cantonese)
        leave_idx = self.extractor.mode % 6  
        charset_idx = self.extractor.mode / 6 + 1  
        if leave_idx != 4:
            self._constructFeatureDict(new_data, 0)
            print 'simplified pair feature size = ', len(self.extractor.featureDict)
        if leave_idx != 5:
            self._constructFeatureDict(new_data, charset_idx)
            print 'pinyin pair feature size = ', len(self.extractor.featureDict)
        #self.extractor.saveFeatureDict('feature.dict')
        x, y = self._extract_features(new_data)
        from sklearn import svm
        from sklearn.feature_extraction import DictVectorizer
        self.vectorizer = DictVectorizer(sparse=True)
        x = self.vectorizer.fit_transform(x)

        clf = svm.LinearSVC(**options)
        clf.fit(x, y)
        self.model = clf


    ''' return a prediction on whether two strings are the same name according to the given model. 
    By default we'll use our pre-trained model 
    '''
    def predict(self, name1, name2, cantonese=False):
        import math
        data = [(1, name1, name2)]
        #print data
        new_data = self._convert_data(data, cantonese)
        x, y = self._extract_features(new_data)

        #need to be fixed
        if len(x) == 0:
            return False, 0.0

        x = self.vectorizer.transform(x)
        # The score of the label for "True"
        score = self.model.decision_function(x)     #predict_proba(x)[0][1]
        prob = math.exp(score) / (math.exp(score)+math.exp(-score))

        return prob > self.threshold, prob 


    ''' save model to file. '''
    def save_model(self, modelfilename, featurefilename):
        import pickle
        with open(modelfilename, 'wb') as file:
            pickle.dump(self.model, file)
            pickle.dump(self.vectorizer, file)
        with open(featurefilename, 'wb') as ffile:
            pickle.dump(self.extractor, ffile)
        #print 'printing feature dict!!!'
        #for k,v in self.extractor.featureDict.items():
        #	print k,v
        #self.extractor.saveFeatureDict(featurefilename)


    def load_feature_extractor(self, featurefilename):
        import pickle
        import features
        with open(featurefilename, 'rb') as ffile:
            self.extractor = pickle.load(ffile)

    def _convert_data(self, data, cantonese=False):
        new_data = []
        for label, name1, name2 in data:
            converted_name_1 = self.character_converter.convert(name1)
            converted_name_2 = self.character_converter.convert(name2)
            #pinyin1_arry = self.pinyinConv.convert(converted_name_1).split()
            #pinyin2_arry = self.pinyinConv.convert(converted_name_2).split()
            #pinyin1 = ''.join(pinyin1_arry)
            #pinyin2 = ''.join(pinyin2_arry)
            pinyin1, pronun1 = self.pronunConv.convert_phoneme(name1)
            if cantonese:
                pinyin2, pronun2 = self.pronunConv.convert_phoneme(name2, mode=1)
            else:
                pinyin2, pronun2 = self.pronunConv.convert_phoneme(name2)
            #pinyin1 = self.pinyinConv.convert(converted_name_1)
            #pinyin2 = self.pinyinConv.convert(converted_name_2)
            if len(pronun1) == 0 or len(pronun2) == 0:
                continue
            new_data.append((label, (converted_name_1, ''.join(pinyin1), pinyin1, pronun1), (converted_name_2, ''.join(pinyin2), pinyin2, pronun2)))
        return new_data


    def _constructNgramidfDict(self, data):
        #return    #NOTICE!!!!!!!!!!!!!!!!!!!!!!!!!!! temporary modification for running experiments
        for label, name1, name2 in data:
            charset_idx = self.extractor.mode / 6 + 1 #7 - 3
            str1 = name1[charset_idx]
            str2 = name2[charset_idx]
            self.extractor.construct_idfDict(name1[0], name2[0])
            self.extractor.construct_idfDict(str1, str2)
        print 'idf dict size = ', len(self.extractor.ngrm_idfDict)


    def _constructFeatureDict(self, data, charset_idx):
        for label, name1, name2 in data:
            if label == False:
                continue
            str1 = name1[charset_idx]
            str2 = name2[charset_idx]
            if self.extractor.align_mode == 'align':
                #print 'constructing aligned pairs!'
                from .similarity import levenshtein_distance
                #self.extractor.constructAlignedFeatureDict(name1[0], name2[0], levenshtein_distance)
                #print 'simplified pair feature size = ', len(self.extractor.featureDict)
                self.extractor.constructAlignedFeatureDict(str1, str2, levenshtein_distance)
                #for item1, item2 in zip(name1, name2):
                #	self.extractor.constructAlignedFeatureDict(item1, item2, levenshtein_distance)
            elif self.extractor.align_mode == 'all':
                #self.extractor.constructPairFeatureDict(name1[0], name2[0])
                #print 'simplified pair feature size = ', len(self.extractor.featureDict)
                self.extractor.constructPairFeatureDict(str1, str2)
                #print 'pinyin pair feature size = ', len(self.extractor.featureDict)
                #for item1, item2 in zip(name1, name2):
                #	self.extractor.constructPairFeatureDict(item1, item2, levenshtein_distance)
            else:
                raise ValueError('Wrong mode for alignment feature extraction! Currently available modes are alignmode and all mode!')
        self.extractor.filter_feature(2)



    ''' generate training/testing data for SVM.'''
    def _extract_features(self, data):
        x = []
        y = []
        #print 'in _extract_features function, len of featureMap =', len(self.extractor.featureDict)
        for label, name1, name2 in data:
            #print label, name1, name2
            features = self.extractor.extract(name1, name2)

            x.append(features)
            if label == True:
                y.append(1)
            else:
                y.append(0)

        #print 'data size =', len(x), len(y)
        return x, y




class FunctionNameMatcher:
    def __init__(self, function, threshold=0.5, character_converter=None, pinyin_converter=None, pronun_converter=None):
        self.function = function
        self.threshold = threshold
        self.charConv = character_converter
        self.pinyinConv = pinyin_converter
        self.pronunConv = pronun_converter
        self.mode = None

    def set_mode(self, mode='original'):
        self.mode = mode

    def set_threshold(self, thre):
        self.threshold = thre

    def predict(self, name1, name2):
        if self.mode == 'simplified':
            name1 = self.charConv.convert(name1)
            name2 = self.charConv.convert(name2)
            #print converted_name_1, '\t', converted_name_2
        elif self.mode == 'pinyin':
            name1 = self.charConv.convert(name1)
            name2 = self.charConv.convert(name2)
            #pinyin1_arry = self.pinyinConv.convert(name1).split()
            #pinyin2_arry = self.pinyinConv.convert(name2).split()
            #name1 = ''.join(pinyin1_arry)
            #name2 = ''.join(pinyin2_arry)
            name1 = self.pinyinConv.convert(name1)
            name2 = self.pinyinConv.convert(name2)
            #print pinyin1, '\t', pinyin2
        elif self.mode == 'pronun1':
            name1, stuff = self.pronunConv.convert_phoneme(name1)
            name2, stuff = self.pronunConv.convert_phoneme(name2)
        elif self.mode == 'pronun2':
            stuff, name1 = self.pronunConv.convert_phoneme(name1)
            stuff, name2  = self.pronunConv.convert_phoneme(name2)
            #Note: pronun3 pronun4 are especially designed for Cantonese
        elif self.mode == 'pronun3':
            name1, stuff = self.pronunConv.convert_phoneme(name1)
            name2, stuff = self.pronunConv.convert_phoneme(name2, mode=1)
            name1 = ''.join(name1)
            name2 = ''.join(name2)
        elif self.mode == 'pronun4':
            stuff, name1 = self.pronunConv.convert_phoneme(name1)
            stuff, name2  = self.pronunConv.convert_phoneme(name2, mode=1)
        elif self.mode != 'original':
            raise ValueError('Invalid name matcher type give: ' + self.mode)
        score = self.function(name1, name2)
        return score > self.threshold, score  #, name1, name2

