'''
Apply a name matcher to a given data set. If labels are included it will evaluate it; else will produce a score for each pair.

Data should be in the following tab separated format:
name1 name2 label(true/false)
'''

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test the name matcher.')
    parser.add_argument('--data', required=True, help='The data on which to run')
    parser.add_argument('--matcher', required=True, help='Which matcher to run.', choices=['classifier', 'jarowinkler', 'levenshtein'])
    parser.add_argument('--data-type', required=True, help='The input data type, choosing from raw or paired, representing the raw matched names or the paired instances with both matching and unmatching pairs, respectively.', choices=['raw', 'paired'])
    parser.add_argument('--output-file', required=False, help='The file in which to store the results of running the matcher. Output will include a label (true/false) and a score')
    parser.add_argument('--threshold', required=False, type=float, help='The threshold to use for the matcher (option for the classifier)')
    parser.add_argument('--model-file', required=False, help='The model file to use for the classifier matcher. If none given, will use the default model')
    parser.add_argument('--feature-map', required=False, help='The orthography feature file to use for the classifier matcher. If none given, will use the empty orthography feature')
    args = parser.parse_args()


    data_file = args.data
    output_file = args.output_file

    threshold = 0.55 
    if args.threshold:
        threshold = args.threshold

    matcher_type = args.matcher
    data_type = args.data_type

    if matcher_type.lower() == 'classifier':
        from .mingpipe import get_chinese_name_matcher, NameMatcher
        from .features import FeatureExtractor
        from .pinyin.langconv import Converter
        from .pinyin.ch2pinyin import Char2Pinyin
        from .pinyin.ch2pronun import Char2Pronun
        from .utils import get_resource

        if args.model_file:
            extractor = FeatureExtractor()
            matcher = NameMatcher(character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')), feature_extractor=extractor, model_filename=args.model_file)
            if args.feature_map:
                #extractor.loadFeatureDict(args.feature_map)
                matcher.load_feature_extractor(args.feature_map)
        else:
            matcher = get_chinese_name_matcher()
    elif matcher_type.lower() == 'jarowinkler':
        from .pinyin.langconv import Converter
        from .pinyin.ch2pinyin import Char2Pinyin
        from .pinyin.ch2pronun import Char2Pronun
        from .utils import get_resource
        from .similarity import jaro_winkler
        from .mingpipe import FunctionNameMatcher
        matcher = FunctionNameMatcher(jaro_winkler, character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')))
    elif matcher_type.lower() == 'levenshtein':
        from .pinyin.langconv import Converter
        from .pinyin.ch2pinyin import Char2Pinyin
        from .pinyin.ch2pronun import Char2Pronun
        from .utils import get_resource
        from .similarity import levenshtein_distance
        from .mingpipe import FunctionNameMatcher
        matcher = FunctionNameMatcher(levenshtein_distance, character_converter=Converter('zh-hans'), pinyin_converter=Char2Pinyin(get_resource('char2pinyin.new')), pronun_converter=Char2Pronun(get_resource('mcpdict.csv')))
    else:
        raise ValueError('Invalid name matcher type give: ' + matcher_type)
    from .utils import load_name_pairs, gen_pairs, save_name_pairs, run_name_matcher, evaluate_f1, evaluate_rank
    #data = load_name_pairs(data_file)
    if data_type == 'paired':
        data = load_name_pairs(data_file)
    elif data_type == 'raw':
        data = gen_pairs(data_file, mode='confuse')
        save_name_pairs(data, data_file+'.paired')
    else:
        raise ValueError('Invalid data type give: ' + data_type)

    if matcher_type.lower() == 'jarowinkler' or matcher_type.lower() == 'levenshtein':
        matcher.set_mode('pronun1')
        #matcher.set_mode('simplified')
        #matcher.set_mode('original')
    
    # hacky part to select threshold.
    '''from .utils import select_threshold
    threshold = select_threshold(data, matcher, num_threshold_levels=20, use_f1=True)
    print 'best threshold:', threshold
    '''
    matcher.set_threshold(threshold)
    
    results = run_name_matcher(matcher, data)
    if data[0][0] != None:
        # This data is labeled. Evaluate it.
        top1, top2, top3, MRR = evaluate_rank(data, results)
        print 'top1 accuracy:', top1, 'top2 accuracy:', top2, 'top3 accuracy:', top3, 'MRR:', MRR
        #prec, recall, F1 = evaluate_f1(data, results)
        #print 'precision:', prec, 'recall:', recall, 'F1:', F1
    '''
    else:
    import codecs as cs
    with cs.open(data_file+'.score', 'w', encoding='utf-8') as outf:
        for (prediction, score), (label, name1, name2) in zip(results, data):
            outf.write(name1+'\t'+name2+'\t'+str(score)+'\n')
    '''

    import codecs
    if output_file:
        with codecs.open(output_file, 'w', 'utf-8') as output:
            for ii, (label, name1, name2) in enumerate(data):
                prediction, score = results[ii]
                output.write('%s\t%s\t%s\t%f\n' % (name1, name2, str(prediction), score))

if __name__ == '__main__':
    main()
