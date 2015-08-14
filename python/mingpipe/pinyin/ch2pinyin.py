# -*- coding: utf-8 -*- 

class Char2Pinyin:
	def __init__(self, filename):
		self.load_dict(filename)

	def load_dict(self, filename):
		self.ch2pinyin = {}
		import codecs
		with codecs.open(filename, 'r', 'utf-8') as input:
			for line in input:
				line = line.strip('\r\n').strip('\n')
				idx = line.find(' ')
				key = line[0:idx]
				val = line[idx+1:]
				self.ch2pinyin[key] = val

		self.maxl = max(map(lambda x: len(x), self.ch2pinyin.keys()))

	def convert(self, chars):
		ret = []
		pos = 0

		while pos < len(chars):
			#if pos != 0:
			#	ret.append(' ')
			find = False
			for i in range(self.maxl, 0, -1):

				if pos + i > len(chars):
					continue
				char = chars[pos: pos+i]

				if char in self.ch2pinyin:
					ret.append(self.ch2pinyin[char])
					pos += i
					find = True

					break
			if not find:
				ret.append(chars[pos])
				pos += 1
		return ''.join(ret)
