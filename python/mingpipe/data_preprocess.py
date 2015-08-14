#!/usr/bin/python
# coding=utf-8 

import sys
import codecs
import re

def split_data(infile, fold):
	data = []
	rpatten = '[\w -]+'
	from .pinyin.langconv import Converter
	character_converter=Converter('zh-hans')
	with codecs.open(infile, 'r', encoding='utf-8') as inf:
		for line in inf:
			skip = False
			elems = line.strip().split('\t')
			if len(elems) > 2:
				continue
			for el in elems:
				if re.sub(rpatten, '', el) == '':
					skip = True
					break
				sel = character_converter.convert(el)
				if sel.endswith(u'列表') or sel.endswith(u'将军') or sel.endswith(u'代表') or sel.endswith(u'运动') or sel.endswith(u'问题') or u'维基' in sel or u'慰安妇' in sel :
					skip = True
					break
			if not skip:
				#print line.rstrip()
				data.append(line)
	#exit(0)
	#block_size = len(data) / fold
	with codecs.open(infile+'.filtered.train', 'w', encoding='utf-8') as trf, codecs.open(infile+'.filtered.dev', 'w', encoding='utf-8') as devf, codecs.open(infile+'.filtered.test', 'w', encoding='utf-8') as tf:
		for i, line in enumerate(data):
			if i % fold < fold - 2:
				trf.write(line)
			elif i % fold == fold - 2:
				devf.write(line)
			else:
				tf.write(line)
	
def error_analysis(infile):
	data = []
	with codecs.open(infile, 'r', encoding='utf-8') as inf:
		for line in inf:
			if line.startswith('Selected threshold'):
				data.append(dict())
			elif line.startswith('error term') :
				elems = line.split()
				data[-1][elems[2]] = line.strip()
		#print len(data)
		sim_dict = data[0]
		pinyin_dict = data[1]
		print 'svm better case:'
		for key in sim_dict:
			if key not in pinyin_dict:
				print  sim_dict[key]
		print 'transducer better case:'
		for key in pinyin_dict:
			if key not in sim_dict:
				print  pinyin_dict[key]
				


if __name__ == '__main__':
	file_name = sys.argv[1]
	split_data(file_name, 5)
	suffix = ['.filtered.train', '.filtered.dev', '.filtered.test']
	from .utils import gen_pairs, save_name_pairs
	for sf in suffix:
		ffname = file_name + sf
		data = gen_pairs(ffname, mode='confuse')
		save_name_pairs(data, ffname+'.paired')
	
	#error_analysis(file_name)
