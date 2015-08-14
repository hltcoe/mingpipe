#!/usr/bin/python
# coding=utf-8

import sys, re, math, random, codecs

def get_resource(resource_name):
	import os
	return os.path.join(os.path.dirname(__file__), 'resources', resource_name)


''' load data from files. 
@param: data_file_name
@param: source_string_array
@param: target_string_array '''
def load_name_pairs(filename):
	data = []
	with codecs.open(filename, 'r', 'utf-8') as input:
		for line in input:
			entries = line.strip().split('\t')
			name1 = entries[0].strip()
			name2 = entries[1].strip()
			label = None
			if len(entries) > 2:
				if entries[2].lower() == 'true':
					label = True
				elif entries[2].lower() == 'false':
					label = False
				else:
					raise ValueError('Label in input file must be true or false. Found: "%s"' % entries[2])
				
			data.append((label, name1, name2))

	return data

def load_traditional_chinese():
    tranSet = set()
    simSet = set()
    import os
    dir = os.path.dirname(__file__)
    with codecs.open(os.path.join(dir, 'pinyin/hanst.dict'), 'r', encoding='utf-8') as inf:
        for line in inf:
            elems = line.strip().split()
            for ele in elems:
                simc, tranc = ele.split('=')
                simSet.add(simc)
                tranSet.add(tranc)
    tranSet -= simSet
    return tranSet

def save_model(model, filename):
	import pickle
	pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
	import pickle
	my_model = pickle.load(open(filename, 'rb'))
	return my_model

def compose_matrix(str1, str2, similarity_handler):
	max_len = max(len(str1), len(str2))
	cost_matrix = []
	for i,char1 in enumerate(str1):
		cost_matrix.append([])
		for j,char2 in enumerate(str2):
			cost_matrix[-1].append(1-similarity_handler(char1, char2))   #+abs(i-j)/max_len)
	return cost_matrix


def alignment(str1, str2, similarity_handler):
	from .munkres import Munkres
	aligner = Munkres()
	cost_matrix = compose_matrix(str1, str2, similarity_handler)
	indexes = aligner.compute(cost_matrix)
	return indexes


'''
def word_to_pronun(word, pronun_dict, mode = 0):
	word_pronun = []
	for w in word:
		if w not in pronun_dict:
			#print w
			continue
		pronun = pronun_dict[w][mode]
		pronun_arry = pronun.split(',')
		word_pronun.append(pronun_arry[0])
	pingyin = ''.join(word_pronun)
	import re
	return re.sub('[0-9]', '', pingyin)


def word_to_pinyin(hanzi):
	from pinyin.langconv import Converter
	from pinyin.ch2pinyin import Char2Pinyin
	hanz_converter = Converter('zh-hans')
	pinyin_converter = Char2Pinyin(get_resource('char2pinyin.new'))
	simplified = hanz_converter.convert(hanzi)
	pinyin_arry = pinyin_converter.convert(simplified) #.split()
	#pinyin_str = ''.join(pinyin_arry)
	return pinyin_arry
'''

''' load the raw data and generate the training/testing pairs.
@param: data_file_name
@param: mode = {random, confuse, all}'''
def gen_pairs(infile_name, mode='random', random_neg=10):
	with codecs.open(infile_name, 'r', encoding='utf-8') as inf:
		tranSet = load_traditional_chinese()
		srcName = []
		targetName = []
		data = []
		#print tranSet
		for line in inf:
			#print line
			elems = line.strip().split()
			if len(elems) > 2:
				continue
			name1, name2 = elems
			name1 = ''.join(name1.split())	
			name2 = ''.join(name2.split())	
			if len(srcName) > 0 and name1 == srcName[-1]:
				continue
			flag = False
			for char_name1 in name1:
				if char_name1 in tranSet:
					flag = True
					break
			if flag:
				srcName.append(name2)
				targetName.append(name1)
				flag = False
			else:
				srcName.append(name1)
				targetName.append(name2)
	if mode == 'random':
		random.seed(1)
		for i, (srcn, targetn) in enumerate(zip(srcName, targetName)):
			tpl = [True, srcn, targetn]
			data.append(tpl)
			#outf.write(' '.join(tpl)+'\n')
			candidate = random.sample(xrange(len(srcName)-1), random_neg)
			for idx in candidate:
				if idx >= i:
					idx+=1
				tpl = [False, srcn, targetName[idx]]
				data.append(tpl)
				#outf.write(' '.join(tpl)+'\n')
	elif mode == 'all':
		for i, (srcn, targetn) in enumerate(zip(srcName, targetName)):
			tpl = [True, srcn, targetn]
			data.append(tpl)
			for j, cand in enumerate(targetName):
				if j == i:
					continue
				tpl = [False, srcn, targetName[j]]
				data.append(tpl)
	elif mode == 'confuse':
		from .similarity import levenshtein_distance
		random.seed(1)
		for i, (srcn, targetn) in enumerate(zip(srcName, targetName)):
			tpl = [True, srcn, targetn]
			data.append(tpl)
			candidate = []
			for j, cand in enumerate(targetName):
				if j == i or srcn == cand or targetn == cand:
					continue
				if levenshtein_distance(srcn, cand) > 0:
					candidate.append(j)
					if len(candidate) >= 10:
						break
			if len(candidate) < 10:
				#print 'irregularity!!', len(candidate)
				sample_list = list(set(xrange(len(srcName))) - set(candidate) - set([i]))
				append_list = random.sample(sample_list, (random_neg-len(candidate)))
				candidate.extend(append_list)
			for el in candidate:
				tpl = [False, srcn, targetName[el]]
				data.append(tpl)
	else:
		raise ValueError('Invalid data construction mode give: ' + mode)
	return data


def save_name_pairs(data, filename):
	with codecs.open(filename, 'w', encoding='utf-8') as outf:
		for dt in data:
			if dt[0] == True:
				label = 'true'
			elif dt[0] == False:  
				label = 'false'
			else:
				raise ValueError('Invalid data label give: ' + label)
			#print label, dt[1], dt[2]
			outf.write('\t'.join(dt[1:]) + '\t' + label + '\n')


def Levenshtein(data, evalResults):
	from .similarity import levenshtein_distance
	for elem in data:
		#print elem
		if elem[0]:
			evalResults.append([])
		evalResults[-1].append((elem[0], levenshtein_distance(elem[1], elem[2])))


def JaroWinkler(data, evalResults):
	from .similarity import jaro_winkler
	for elem in data:
		#print elem
		if elem[0] : 
			evalResults.append([])
		evalResults[-1].append((elem[0], jaro_winkler(elem[1], elem[2])))


def appendNgramOverlapFeat_bin(featString, source, target, begin, simArray):
	sourceNgram = set()
	targetUni = set()
	targetBi = set()
	targetTri = set()
	for i in range(len(source)):
		sourceNgram.add(source[i])
		if i < len(source) - 1:
			sourceNgram.add(source[i] + ' ' + source[i+1] )
		if i < len(source) - 2:
			sourceNgram.add(source[i] + ' ' + source[i+1] + ' ' + source[i+2] )
	unigram = 0
	bigram = 0
	trigram = 0
	for i in range(len(target)):
		targetUni.add(target[i])
		if i < len(target) - 1:
			targetBi.add(target[i] + ' ' + target[i+1] )
		if i < len(target) - 2:
			targetTri.add(target[i] + ' ' + target[i+1] + ' ' + target[i+2] )
	unigram = len(targetUni & sourceNgram)
	bigram = len(targetBi & sourceNgram)
	trigram = len(targetTri & sourceNgram)
	featureArray = []
	if unigram != 0:
		tempSim = float(unigram)/math.sqrt(len(targetUni))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureArray.append(':'.join( (str(begin+i), '1')))
				#featString += ' ' + str(begin+i) + ':1' #+ str(float(unigram)/math.sqrt(len(targetUni)))
	if bigram != 0:
		tempSim = float(bigram)/math.sqrt(len(targetBi))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureArray.append(':'.join( (str(begin+len(simArray)+i), '1')))
				#featString += ' ' + str(begin+len(simArray)+i) + ':1' #+ str(float(bigram)/math.sqrt(len(targetBi)))
	if trigram != 0:
		tempSim = float(trigram)/math.sqrt(len(targetTri))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureArray.append(':'.join( (str(begin+2*len(simArray)+i), '1')))
				#featString += ' '+ str(begin+2*len(simArray)+i) + ':1' #+ str(float(trigram)/math.sqrt(len(targetTri)))
	featString += ' ' + ' '.join(featureArray)
	return featString


def appendNgramOverlapFeat_bin_ch(featureMap, source, target, begin, simArray):
	sourceNgram_p = set()
	targetUni_p = set()
	targetBi_p = set()
	targetTri_p = set()
	sourceNgram_c = set()
	targetUni_c = set()
	targetBi_c = set()
	srcElem = source.split(':')
	for i in range(len(srcElem[0])):
		sourceNgram_p.add(srcElem[0][i])
		if i < len(srcElem[0]) - 1:
			sourceNgram_p.add(srcElem[0][i] + ' ' + srcElem[0][i+1] )
		if i < len(srcElem[0]) - 2:
			sourceNgram_p.add(srcElem[0][i] + ' ' + srcElem[0][i+1] + ' ' + srcElem[0][i+2] )
	unigram = 0
	bigram = 0
	trigram = 0
	tarElem = target.split(':')
	for i in range(len(tarElem[0])):
		targetUni_p.add(tarElem[0][i])
		if i < len(tarElem[0]) - 1:
			targetBi_p.add(tarElem[0][i] + ' ' + tarElem[0][i+1] )
		if i < len(tarElem[0]) - 2:
			targetTri_p.add(tarElem[0][i] + ' ' + tarElem[0][i+1] + ' ' + tarElem[0][i+2] )
	unigram = len(targetUni_p & sourceNgram_p)
	bigram = len(targetBi_p & sourceNgram_p)
	trigram = len(targetTri_p & sourceNgram_p)
#	featureMap = {}
	if unigram != 0:
		tempSim = float(unigram)/math.sqrt(len(targetUni_p))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureMap[begin+i] = 1
				#featString += ' ' + str(begin+i) + ':1' #+ str(float(unigram)/math.sqrt(len(targetUni_p)))
	if bigram != 0:
		tempSim = float(bigram)/math.sqrt(len(targetBi_p))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureMap[begin+len(simArray)+i] = 1
				#featureArray.append(':'.join( (str(begin+len(simArray)+i), '1')))
				#featString += ' ' + str(begin+len(simArray)+i) + ':1' #+ str(float(bigram)/math.sqrt(len(targetBi_p)))
	if trigram != 0:
		tempSim = float(trigram)/math.sqrt(len(targetTri_p))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureMap[begin+2*len(simArray)+i] = 1
				#featureArray.append(':'.join( (str(begin+2*len(simArray)+i), '1')))
				#featString += ' '+ str(begin+2*len(simArray)+i) + ':1' #+ str(float(trigram)/math.sqrt(len(targetTri_p)))
	# Generate Chinese charactor features
	for i in range(len(srcElem[1])):
		sourceNgram_c.add(srcElem[1][i])
		if i < len(srcElem[1]) - 1:
			sourceNgram_c.add(srcElem[1][i] + ' ' + srcElem[1][i+1] )
	unigram = 0
	bigram = 0
	for i in range(len(tarElem[1])):
		targetUni_c.add(tarElem[1][i])
		if i < len(tarElem[1]) - 1:
			targetBi_c.add(tarElem[1][i] + ' ' + tarElem[1][i+1] )
	unigram = len(targetUni_c & sourceNgram_c)
	bigram = len(targetBi_c & sourceNgram_c)
#	featureMap = {}
	if unigram != 0:
		tempSim = float(unigram)/math.sqrt(len(targetUni_c))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureMap[begin+3*len(simArray)+i] = 1
				#featString += ' ' + str(begin+i) + ':1' #+ str(float(unigram)/math.sqrt(len(targetUni_c)))
	else :
		featureMap[begin+5*len(simArray)] = 1
	if bigram != 0:
		tempSim = float(bigram)/math.sqrt(len(targetBi_c))
		for i in range(len(simArray)):
			if tempSim > simArray[i]:
				featureMap[begin+4*len(simArray)+i] = 1
				#featureArray.append(':'.join( (str(begin+len(simArray)+i), '1')))
				#featString += ' ' + str(begin+len(simArray)+i) + ':1' #+ str(float(bigram)/math.sqrt(len(targetBi_c)))


def appendStringDist_bin(featString, sim, begin, simArray):
	featureArray = []
	for i in range(len(simArray)):
		if sim > simArray[i]:
			featureArray.append(':'.join( (str(begin+i), '1')))
	featString += ' ' + ' '.join(featureArray)
	return featString


def generateFeatureFile(filename, data, simArray):
	with open(filename, 'w', 'utf-8') as outf:
		for line in data:
			string = line[0]
			string = appendNgramOverlapFeat_bin(string, line[1], line[2], 1, simArray)
			outf.write(string + '\n')


def generateFeatureFile_cv(fileprefix, data, simArray, fold):
	from .similarity import levenshtein_distance
	trainFileArray = []
	testFileArray = []
	for i in range(fold):
		trainFileArray.append(open(fileprefix+'.train.'+str(i), 'w'))
		testFileArray.append(open(fileprefix+'.test.'+str(i), 'w'))
	testSize = len(data)/11/fold
	for i in range(fold):
		for line in data[(i*testSize*11):((i+1)*testSize*11)] :
			string = line[0]
			string = appendNgramOverlapFeat_bin(string, line[1], line[2], 1, simArray)
			#string = appendStringDist_bin(string, jaro_winkler(line[1], line[2]), 1+3*len(simArray), simArray)
			string = appendStringDist_bin(string, float(levenshtein_distance(line[1], line[2]))/min(len(line[1]), len(line[2])), 1+3*len(simArray), simArray)
			for f in trainFileArray[0:i]:
				f.write(string + '\n')
			for f in trainFileArray[(i+1):]:
				f.write(string + '\n')
			testFileArray[i].write(string + '\n')

	for f in trainFileArray:
		f.close()
	for f in testFileArray:
		f.close()


# generate train/test data for svm 
def generateDataforSVM(data, simArray, x, y) :
	for line in data:
		#print line
		y.append( int(line[0]) )
		localFeature = {}
		appendNgramOverlapFeat_bin_ch(localFeature, line[1], line[2], 1, simArray)
		x.append(localFeature )
		#print localFeature
		#print x
		#return (x, y)


# result array, top k, whether reverse the sorting.
def evaluation(evalResults, k, r):
    topK_hit = 0
    RR = 0.0
    notInTop = 0
    for row in evalResults:
        row.sort(key=lambda x: float(x[1]), reverse=r) 
        for i in range(k):
            if row[i][0] :
                topK_hit += 1
                RR += 1.0/(i+1)
                break
        if (i == k-1) and ( not row[i][0]):
            notInTop += 1
            for i in range(k, len(row)):
                if row[i][0] :
                    RR += 1.0/(i+1)
                    break
    print 'TOP'+str(k)+':', float(topK_hit) / len(evalResults)
    print 'RR:', RR / len(evalResults), RR 
    print 'Not in top hits:', notInTop


# result array, error array(for error analysis), top k, whether reverse the sorting.
def evaluation_threshold(evalResults, errArray, r, threshold):
	RR = 0.0
	notInTop = 0
	count = 0
	aggr_prec = 0.0
	aggr_recall = 0.0
	for row in evalResults:
		#print row
		hit = 0.0
		row.sort(key=lambda x: float(x[1]), reverse=r) 
		#print row
		for i in range(len(row)):
			if r and float(row[i][1]) < threshold :
				break
			elif not r and float(row[i][1]) > threshold :
				break
			if row[i][0] :
				hit = 1.0
				RR += 1.0/(i+1)
		prec = hit/1
		recall = hit/(i+1)
		aggr_prec += prec
		aggr_recall += recall
		if hit == 0.0 :
			notInTop += 1
			for j in range(i, len(row)):
				if row[j][0] :
					RR += 1.0/(j+1)
					errArray.append([str(count), ':'.join(row[j][2:]), ':'.join(row[0][2:])])
					break
		count += 1
	prec = aggr_prec / len(evalResults)
	recall = aggr_recall / len(evalResults)
	print 'PRECISION:', aggr_prec / len(evalResults)
	print 'RECALL:', aggr_recall / len(evalResults)
	if prec+recall != 0:
		f1 = 2*prec*recall/(prec+recall)
	else:
		f1 = 0
	print 'F1:', f1
	print 'RR:', RR / len(evalResults), RR 
	print 'Not in top hits:', notInTop


def run_name_matcher(matcher, data):
	results = []
	for ii, (label, name1, name2) in enumerate(data):
		#print name1, name2
		prediction, score = matcher.predict(name1, name2)
		results.append((prediction, score))

	return results

def evaluate_rank(data, results):
	topK_hits = [0.0, 0.0, 0.0]
	Ks = [1.0, 2.0, 3.0]
	RR = 0.0
	notInTop = 0
	groupMap = {}
	from random import shuffle
	#from similarity import levenshtein_distance
	for ii, (label, name1, name2) in enumerate(data):
		#if len(results[ii]) > 2:
		#	name2 = results[ii][-1]
		if name1 in groupMap:
			#groupMap[name1].append((name2, results[ii][-2], label, results[ii][0], levenshtein_distance(name1, name2), results[ii][1]))
			groupMap[name1].append((name2, label, results[ii][0], results[ii][1]))
		else:
			groupMap[name1] = [(name2, label, results[ii][0], results[ii][1])]
			#groupMap[name1] = [(name2, results[ii][-2], label, results[ii][0], levenshtein_distance(name1, name2), results[ii][1])]
	#top1_array = []
	#RR_array = []
	RR_var = 0.0
	topK_vars = [0.0, 0.0, 0.0]
	for k, val in groupMap.items():
		val.sort(key=lambda x: float(x[-1]), reverse=True) 
		val_str = [itm[0] for itm in val]
		#print k, [k], val
		#print '; '.join(val_str)
		if val[0][-1] == 0 or val[0][-1] == val[-1][-1]:
			#RR += 1.0/(len(val))
			#continue
			#shuffle(val)
			print 'cannot compare!!!!'
			print 'error term:', k, ', result rank: [', ' '.join([elem[0] for elem in val]), '].', val
			RR += 0.2745
			RR_var += pow(0.2745, 2)
			#RR_array.append(0.293)
			for ik, vk in enumerate(Ks):
				topK_hits[ik] += Ks[ik] / 10
			#print 'tied! MRR = ', RR, 'topK = ', topK_hits
		else :
		    for i, (nstr, lb, pred, scr) in enumerate(val):
			if lb == True:
				if i != 0:
					print 'error term:', k, ', result rank: [', ' '.join([elem[0] for elem in val]), ']. ground truth:', nstr, ', score:', scr
				for ik, vk in enumerate(Ks):
					if i < vk:
						topK_hits[ik] += 1
						topK_vars[ik] += 1
				RR += 1.0/(i+1)
				RR_var += pow(1.0/(i+1), 2)
				#RR_array.append(1.0/(i+1))
				break	
		    #print 'MRR = ', RR, 'topK = ', topK_hits
	#print 'TOP'+str(k)+':', float(topK_hit) / len(evalResults)
	#print 'RR:', RR / len(evalResults), RR 
	#print 'Not in top hits:', notInTop
	print 'topk variances:'
	for i in range(3):
		print float(topK_vars[i])/len(groupMap) - pow(topK_hits[i]/len(groupMap), 2)
	print 'RR variance:', RR_var/len(groupMap) - pow(RR/len(groupMap), 2)
	return float(topK_hits[0])/len(groupMap), float(topK_hits[1])/len(groupMap), float(topK_hits[2])/len(groupMap), RR/len(groupMap) 


def evaluate_f1(data, results):
	predicted_pos = 0
	true_pos = 0
	groupMap = {}
	for ii, (label, name1, name2) in enumerate(data):
		if name1 in groupMap:
			groupMap[name1].append((name2, label, results[ii][0], results[ii][1]))
		else:
			groupMap[name1] = [(name2, label, results[ii][0], results[ii][1])]
	for k, val in groupMap.items():
		val.sort(key=lambda x: float(x[-1]), reverse=True) 
		val_str = [itm[0] for itm in val]
		#print k, [k], val
		#print '; '.join(val_str)
		for itm in val:
			name2, lb, prediction, score = itm 
			if prediction == False:
				break
			elif lb == True:
				true_pos += 1
			predicted_pos += 1
		#print 'predicted positive = ', predicted_pos, 'true positive = ', true_pos	
	if true_pos + predicted_pos == 0:
		precision = 0
	else:
		precision = float(true_pos) / predicted_pos 
	recall = float(true_pos) / len(groupMap) 
	if precision + recall != 0:
		f1 = float(2 * precision * recall) / float(precision + recall)
	else:
		f1 = 0

	#accuracy = num_correct / float(total)
	
	return precision, recall, f1
		
def select_threshold(data, matcher, num_threshold_levels=10, use_f1=False):
	'''
	Select a threshold that gives the best performance on the given dataset.
	num_threshold_levels says how many different thresholds to test, from 0 to the maximum 
	score in the data.
	use_f1 Judge the best score by the F1. Otherwise use accuracy.
	'''
	results = run_name_matcher(matcher, data)
	max_score = 0
	for prediction, score in results:
		max_score = max(score, max_score)
	
	#print 'maximum score = ', max_score, num_threshold_levels
	step = float(max_score) / num_threshold_levels
	threshold = 0
	best_score = None
	best_threshold = None
	#matcher.threshold = step
	
	for ii in range(num_threshold_levels):
		threshold += step
		
		new_results = []
		for prediction, score in results:
			if score > threshold:
				prediction = True
			else:
				prediction = False
			new_results.append((prediction, score))
			
		precision, recall, f1 = evaluate_f1(data, new_results)
	
		current_score = precision 
		if use_f1:
			current_score = f1
		
		if best_score == None or best_score < current_score:
			best_score = current_score
			best_threshold = threshold
			#print 'current best threshold = ', threshold
	
	return best_threshold

