# -*- coding: utf-8 -*- 

class Char2Pronun:
	def __init__(self, filename):
		self.load_dict(filename)
		self.phoneme = set(['b','p','m','f','d','t','n','l','g','k','h','j','q','x','z','c','s','r','zh','ch','sh','y','w','a','o','e','i','u','v','ai','ei','ui','ao','ou','iu','ie','ve','er','an','en','in','un','ang','eng','ing','ong'])
		self.vowel = set(['a', 'o', 'e', 'i', 'u', 'v', 'y'])
		self.cantCons = set(['gw', 'ct', 'ng', 'c', 'b', 'd', 'g', 'f', 'h', 'k', 'j', 'm', 'l', 'n', 'p', 's', 'kw', 't', 'w', 'z'])
		self.hanCons = set(['b','p','m','f','d','t','n','l','g','k','h','j','q','x','z','c','s','r','zh','ch','sh','y','w'])
		self.shmax = 2
		self.yvmax=3

	def load_dict(self, infile):
		self.pronun_dict = {}
		csvf = self.csv_unireader(open(infile))
		for row in csvf:#spamreader: #unicode_csvf[:5]:
			#print row[0]
			self.pronun_dict[row[0]] = row[3:5]
				#print row[3:5]
	
	def canto_stats(self):
		consonant_set = set()
		for k, v in self.pronun_dict.items():
			cant_pronun = v[1]
			if len(cant_pronun) == 0:
				continue
			if cant_pronun[1] not in self.vowel:
				consonant_set.add(cant_pronun[:2])
			else:
				consonant_set.add(cant_pronun[0])
		for charac in consonant_set:
			print charac

	
	def csv_unireader(self, f, encoding="utf-8"):
		import codecs
		import csv
		for row in csv.reader(codecs.iterencode(codecs.iterdecode(f, encoding), "utf-8")):
			yield [e.decode("utf-8") for e in row]


	def convert(self, chars, mode=0):
		word_pronun = []
		for w in chars:
			if w not in self.pronun_dict:
				#print w
				continue
			pronun = self.pronun_dict[w][mode]
			pronun_arry = pronun.split(',')
			word_pronun.append(pronun_arry[0])
		pingyin = ''.join(word_pronun)
		import re
		return re.sub('[0-9]', '', pingyin)


	
	def convert_phoneme(self, chars, mode=0):
		word_pinyin = []
		word_pronun = []
		import re
		for w in chars:
			if w not in self.pronun_dict:
				#print w
				continue
			pronun = self.pronun_dict[w][mode]
			if len(pronun) == 0:
				pronun = self.pronun_dict[w][0]
			pronun_arry = re.split('\W+', pronun)#pronun.split(',')
			phonemes = self.parse_phoneme(pronun_arry[0][:-1], mode)
			word_pinyin.append(pronun_arry[0][:-1])
			word_pronun.extend(phonemes)
		return tuple(word_pinyin), tuple(word_pronun)


	def parse_phoneme(self, str, mode=0):
		phonemes = []
		if mode == 0:
			char_set = self.hanCons
		elif mode == 1:
			char_set = self.cantCons
		if str[:2] in char_set:  #self.phoneme:
			phonemes.append(str[:2])
			str = str[2:]
			if len(str) != 0:
				phonemes.append(str)
		else:
			phonemes.append(str[0])
			str = str[1:]
			if len(str) != 0:
				phonemes.append(str)
		'''for i in range(3,0,-1):
			if str[-i:] in self.phoneme:
				if i < len(str):
					phonemes.append(str[:-i])
				phonemes.append(str[-i:])
				break'''
		return phonemes


if __name__ == '__main__':
	my_converter = Char2Pronun('resources/mcpdict.csv')
	#my_converter.canto_stats()
	import codecs as cs
	import sys
	with cs.open(sys.argv[1], encoding='utf-8') as inf:
		for line in inf:
			name1, name2 = line.rstrip().split()
			phonetic = my_converter.convert_phoneme(name2, mode=1)
			print phonetic 

