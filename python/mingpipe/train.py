'''
Train a new name matcher.

The name matcher requires data to be in the following format:
name1 name2 label(true/false)

If you only have lists of paired names you can create data in this format, including negative examples
by using data.py

@author Nanyun Peng
'''

def train_classifier(train_datafile, dev_datafile, model_file, feature_file, data_type, feature_mode, align_type, cantonese, tune_parameters, cost_coeffecient=0.0000625, pos_weight=4, select_threshold=False, threshold=0.5):
    from .utils import load_name_pairs, gen_pairs, save_name_pairs, evaluate_f1, run_name_matcher, evaluate_rank
    if data_type == 'paired':
        train_data = load_name_pairs(train_datafile)
        dev_data = load_name_pairs(dev_datafile)
    elif data_type == 'raw':
        train_data = gen_pairs(train_datafile, mode='confuse')
        dev_data = gen_pairs(dev_datafile, mode='confuse')
        save_name_pairs(train_data, train_datafile+'.paired')
        save_name_pairs(dev_data, dev_datafile+'.paired')
    else:
        raise ValueError('Invalid data type give: ' + data_type)

    print 'Loaded %d training examples.' % len(train_data)
    print 'Loaded %d development examples.' % len(dev_data)

    from .mingpipe import NameMatcher
    from .features import FeatureExtractor
    from .pinyin.langconv import Converter
    from .pinyin.ch2pinyin import Char2Pinyin
    from .pinyin.ch2pronun import Char2Pronun
    from .utils import get_resource, select_threshold
    extractor = FeatureExtractor(mode=feature_mode, align_mode=align_type)	
    matcher = NameMatcher(character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')), feature_extractor=extractor)
    max_score = 0.0
    max_params = [cost_coeffecient, pos_weight]
    ''' for parameter tuning. If you don't want to tune
    parameters, skip this part. give your choice if -c and -w1 to train.
    eg: matcher.train(train_data, '-c 0.0000625 -w1 4', cantonese)'''
    if tune_parameters:
        for i in range(-4, 5):
            c = cost_coeffecient * pow(5, i) 
            for j in range(5):	
                w = 2 + j*2
                print 'try the parameter', c, w
                matcher.train(train_data, options={'C':c, 'class_weight':{1:w}}, cantonese=cantonese)
                results = run_name_matcher(matcher, dev_data)
                top1, top2, top3, MRR = evaluate_rank(dev_data, results)
                if MRR > max_score:
                    max_score = MRR
                    max_params[0] = c
                    max_params[1] = w
                    print 'current best parameters are:', c, w, ', top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
                    results = run_name_matcher(matcher, train_data)
                    top1, top2, top3, MRR = evaluate_rank(train_data, results)
                    print 'The performance on training data are --- top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
    ''' for parameter tuning. If you don't want to tune
    parameters, skip this part.'''
    print 'best parameters: C =', max_params[0], ', w1 =', max_params[1]  
    matcher.train(train_data, options={'C':max_params[0], 'class_weight':{1:max_params[1]}}, cantonese=cantonese)
    if model_file and feature_file:
        print 'Saving model to: ', model_file
        print 'Saving feature map to: ', feature_file
        matcher.save_model(model_file,feature_file)
   
    if select_threshold:
        threshold = select_threshold(dev_data, matcher, num_threshold_levels=20, use_f1=True)
    print 'the best threshold for positive pairs is:', threshold
    matcher.set_threshold(threshold)
    print 'Training data performance :'
    results = run_name_matcher(matcher, train_data)
    top1, top2, top3, MRR = evaluate_rank(train_data, results)
    print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR

    print 'Development data performance :' 
    results = run_name_matcher(matcher, dev_data)
    top1, top2, top3, MRR = evaluate_rank(dev_data, results)
    print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train the name matcher.')
    parser.add_argument('--train-data', required=True, help='The data on which to train the model')
    parser.add_argument('--dev-data', required=False, help='The data on which to tune model hyper-parameters. Without this the training data will be used')
    parser.add_argument('--data-type', required=True, help='The input data type, choosing from raw or paired, representing the raw matched names or the paired instances with both matching and unmatching pairs, respectively', choices=['raw', 'paired'])
    parser.add_argument('--matcher', required=True, help='Which matcher to run.', choices=['classifier', 'jarowinkler', 'levenshtein'])
    parser.add_argument('--model-file', required=False, help='The place to save a trained model for the classifier matcher.')
    parser.add_argument('--feature-map', required=False, help='The place to save a targeted orthography features for the classifier matcher.')
    parser.add_argument('--feature-mode', required=False, help='What sets of features to use in the learning based model.')
    parser.add_argument('--align-mode', required=False, help='What type string-pair features to use.', choices=['align', 'all'], default='all')
    parser.add_argument('--tune-parameters', required=False, help='Whether to tune parameters during training.', choices=['true', 'false'], default='false')
    parser.add_argument('--select-threshold', required=False, help='Whether to tune the threshold for postive examples.', choices=['true', 'false'], default='false')
    parser.add_argument('--threshold', required=False, help='Threshold for tagging an instance as a positive example.', default=0.5)
    parser.add_argument('--C', required=False, help='The coefficient parameter for the error term C.', default=0.000625)
    parser.add_argument('--w1', required=False, help='The weight for positive examples.', default=4)
    parser.add_argument('--contains-cantonese', required=False, help='Does one of the names containes cantonese pronunciation.', choices=['true', 'false'])

    args = parser.parse_args()


    train_data = args.train_data
    dev_data = args.dev_data
    if dev_data == None:
        dev_data = train_data

    matcher_type = args.matcher
    data_type = args.data_type
    feature_mode = args.feature_mode
    align_type = args.align_mode
    is_threshold_function = False
    cantonese = False
    if args.contains_cantonese and args.contains_cantonese == 'true':
        print 'contains cantonese!!!'
        cantonese = True
    tune_parameters = False
    if args.tune_parameters and args.tune_parameters == 'true':
        tune_parameters = True
    C = float(args.C)
    w1 = float(args.w1)
    threshold = float(args.threshold)
    select_threshold = (args.select_threshold == 'true')

    if matcher_type.lower() == 'classifier':
        train_classifier(train_data, dev_data, args.model_file, args.feature_map, data_type, feature_mode, align_type, cantonese, tune_parameters, C, w1, select_threshold, threshold)

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
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the original character set: %f' % threshold
        # On original character
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the original characters:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
        print '-----------------------------------------------------------'	
        matcher.set_mode('simplified')
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the simplified character set: %f' % threshold
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the simplified characters:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
        print '-----------------------------------------------------------'	
        matcher.set_mode('pinyin')
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the pinyin character set: %f' % threshold
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the pinyin:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
        print '-----------------------------------------------------------'	
        matcher.set_mode('pronun1')
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the pronunciation character set: %f' % threshold
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the pronunciation transcript:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
        print '-----------------------------------------------------------'	
        matcher.set_mode('pronun2')
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the spelling character set: %f' % threshold
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the spelling transcript:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
        print '-----------------------------------------------------------'	
        matcher.set_mode('pronun3')
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the spelling character set: %f' % threshold
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the cantonese transcript:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR

        print '-----------------------------------------------------------'	
        matcher.set_mode('pronun4')
        if select_threshold:
            threshold = select_threshold(data, matcher, num_threshold_levels=100, use_f1=True)
        print 'Selected threshold on the spelling character set: %f' % threshold
        matcher.set_threshold(threshold)
        results = run_name_matcher(matcher, data)
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'The results on the cantonese spelling:'
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR

    print 'Done'

if __name__ == '__main__':
    main()
