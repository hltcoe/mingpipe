'''
Train a new name matcher.

The name matcher requires data to be in the following format:
name1 name2 label(true/false)

If you only have lists of paired names you can create data in this format, including negative examples
by using data.py
'''

def cross_validation(train_datafile, model_file, feature_file, model_type, data_type, feature_mode, align_type, cantonese):
	from .utils import load_name_pairs, gen_pairs, save_name_pairs, evaluate_f1, run_name_matcher, evaluate_rank
	if data_type == 'paired':
		train_data = load_name_pairs(train_datafile)
	elif data_type == 'raw':
		train_data = gen_pairs(train_datafile, mode='confuse')
		save_name_pairs(train_data, train_datafile+'.paired')
	else:
		raise ValueError('Invalid data type give: ' + data_type)
	
	print 'Loaded %d training examples.' % len(train_data)
		
	from .mingpipe import NameMatcher
	from .features import FeatureExtractor
	from .pinyin.langconv import Converter
	from .pinyin.ch2pinyin import Char2Pinyin
	from .pinyin.ch2pronun import Char2Pronun
	from .utils import get_resource, select_threshold
	extractor = FeatureExtractor(mode=feature_mode, align_mode=align_type)
	
	matcher = NameMatcher(character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')), feature_extractor=extractor)		
	'''c = 0.001
	w = 6
	AUC = matcher.cross_valid(train_data, '-c '+ str(c) + ' -w1 '+str(w), 5, cantonese)
    	print 'try the parameter', c, w, ', auc =', AUC
	exit(0)
	'''
	best_score = 0 #float('Inf')
	best_params = [0.0, 0.0]
	for i in range(2, 11):
	    c = 0.0000001 * pow(5, i) 
	    for j in range(5):	
		w = 2 + j*2
		#matcher.train(train_data, '-c '+ str(c) + ' -w1 '+str(w), cantonese)
		AUC = matcher.cross_valid(train_data, '-c '+ str(c) + ' -w1 '+str(w), 5, cantonese)
	    	print 'try the parameter', c, w, ', auc =', AUC
		if AUC > best_score:
			best_score = AUC 
			best_params[0] = c
			best_params[1] = w
			print 'current best parameters are:', c, w, ', AUC:', AUC 
	if model_file and feature_file:
		matcher.train(train_data, '-c '+ str(best_params[0]) + ' -w1 '+str(best_params[1]), cantonese)
		print 'Saving model to: ', model_file
		print 'Saving feature map to: ', feature_file
		matcher.save_model(model_file,feature_file)


def main():
	import argparse
	
	parser = argparse.ArgumentParser(description='Train the name matcher.')
	parser.add_argument('--train-data', required=True, help='The data on which to train the model')
	parser.add_argument('--data-type', required=True, help='The input data type, choosing from raw or paired, representing the raw matched names or the paired instances with both matching and unmatching pairs, respectively', choices=['raw', 'paired'])
	parser.add_argument('--matcher', required=True, help='Which matcher to run.', choices=['classifier', 'jarowinkler', 'levenshtein'])
	parser.add_argument('--model-file', required=False, help='The place to save a trained model for the classifier matcher.')
	parser.add_argument('--feature-map', required=False, help='The place to save a targeted orthography features for the classifier matcher.')
	parser.add_argument('--feature-mode', required=False, help='What sets of features to use in the learning based model.')
	parser.add_argument('--align-mode', required=False, help='What type string-pair features to use.', choices=['align', 'all'], default='all')
	parser.add_argument('--contains-cantonese', required=False, help='Does one of the names containes cantonese pronunciation.', choices=['true', 'false'])

	args = parser.parse_args()
	
	
	train_data = args.train_data

	model_type = args.model_type
	matcher_type = args.matcher
	data_type = args.data_type
	feature_mode = args.feature_mode
	align_type = args.align_mode
	is_threshold_function = False
	cantonese = False
	if args.contains_cantonese and args.contains_cantonese == 'true':
		print 'contains cantonese!!!'
		cantonese = True


	if matcher_type.lower() == 'classifier':
		cross_validation(train_data, args.model_file, args.feature_map,  model_type, data_type, feature_mode, align_type, cantonese)
		#is_threshold_function = True	
		
	elif matcher_type.lower() == 'jarowinkler':
		from .similarity import jaro_winkler
		from .mingpipe import FunctionNameMatcher
		from .utils import load_name_pairs, gen_pairs, save_name_pairs
		from .pinyin.langconv import Converter
		from .pinyin.ch2pinyin import Char2Pinyin
		from .pinyin.ch2pronun import Char2Pronun
		from .utils import get_resource
		if data_type == 'paired':
			data = load_name_pairs(train_data)
		elif data_type == 'raw':
			data = gen_pairs(train_data, mode='confuse')
			save_name_pairs(data, train_data+'.paired')
		else:
			raise ValueError('Invalid data type give: ' + data_type)
		matcher = FunctionNameMatcher(jaro_winkler, character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')))
		is_threshold_function = True
		
	elif matcher_type.lower() == 'levenshtein':
		from .similarity import levenshtein_distance
		from .mingpipe import FunctionNameMatcher
		from .utils import load_name_pairs, gen_pairs, save_name_pairs
		from .pinyin.langconv import Converter
		from .pinyin.ch2pinyin import Char2Pinyin
		from .pinyin.ch2pronun import Char2Pronun
		from .utils import get_resource
		if data_type == 'paired':
			data = load_name_pairs(train_data)
		elif data_type == 'raw':
			data = gen_pairs(train_data, mode='confuse')
			#print data
			save_name_pairs(data, train_data+'.paired')
		else:
			raise ValueError('Invalid data type give: ' + data_type)
		matcher = FunctionNameMatcher(levenshtein_distance, character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')))
		is_threshold_function = True
	else:
		raise ValueError('Invalid name matcher type give: ' + matcher_type)
	
	if is_threshold_function:	
		from .utils import evaluate_f1, run_name_matcher, evaluate_rank, select_threshold
		matcher.set_mode('original')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected threshold on the original character set: %f' % threshold
		# On original character
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		#accuracy, precision, recall, f1 = evaluate(data, results)
		#print 'Accuracy: %f, Precision: %f, Recall: %f, F1: %f' % (accuracy, precision, recall, f1)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the original characters:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
		#prec, recall, F1 = evaluate_f1(data, results)
		#print 'precision:', prec, 'recall:', recall, 'F1:', F1
		print '-----------------------------------------------------------'	
		matcher.set_mode('simplified')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected threshold on the simplified character set: %f' % threshold
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the simplified characters:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
		#prec, recall, F1 = evaluate_f1(data, results)
		#print 'precision:', prec, 'recall:', recall, 'F1:', F1
		print '-----------------------------------------------------------'	
		matcher.set_mode('pinyin')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected threshold on the pinyin character set: %f' % threshold
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the pinyin:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
		#prec, recall, F1 = evaluate_f1(data, results)
		#print 'precision:', prec, 'recall:', recall, 'F1:', F1
		print '-----------------------------------------------------------'	
		matcher.set_mode('pronun1')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected threshold on the pronunciation character set: %f' % threshold
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the pronunciation transcript:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
		#prec, recall, F1 = evaluate_f1(data, results)
		#print 'precision:', prec, 'recall:', recall, 'F1:', F1
		print '-----------------------------------------------------------'	
		matcher.set_mode('pronun2')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected thresho2d on the spelling character set: %f' % threshold
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the spelling transcript:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
		#prec, recall, F1 = evaluate_f1(data, results)
		#print 'precision:', prec, 'recall:', recall, 'F1:', F1
		print '-----------------------------------------------------------'	
		matcher.set_mode('pronun3')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected thresho2d on the spelling character set: %f' % threshold
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the cantonese transcript:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
		
		print '-----------------------------------------------------------'	
		matcher.set_mode('pronun4')
		threshold = 0.5 #select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
		print 'Selected thresho2d on the spelling character set: %f' % threshold
		matcher.set_threshold(threshold)
		results = run_name_matcher(matcher, data)
		top1, top2, top3, MRR = evaluate_rank(data, results)
		print 'The results on the cantonese spelling:'
		print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR

	print 'Done'

if __name__ == '__main__':
	main()
