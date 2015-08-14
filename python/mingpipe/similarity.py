# The copyright statement from Jellyfish
'''
Copyright (c) 2010, Sunlight Labs

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, 
      this list of conditions and the following disclaimer in the documentation 
      and/or other materials provided with the distribution.
    * Neither the name of Sunlight Labs nor the names of its contributors may be
      used to endorse or promote products derived from this software without 
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Copied from Jellyfish (https://pypi.python.org/pypi/jellyfish/0.3.2)
def levenshtein_distance(s1, s2, damerau=False):
	if s1 == s2:
		return 1
	rows = len(s1)+1
	cols = len(s2)+1

	if not s1:
		return 1-float(cols-1)/cols
	if not s2:
		return 1-float(rows-1)/rows

	prev = None
	cur = range(cols)
	for r in xrange(1, rows):
		prevprev, prev, cur = prev, cur, [r] + [0]*(cols-1)
		for c in xrange(1, cols):
			deletion = prev[c] + 1
			insertion = cur[c-1] + 1
			edit = prev[c-1] + (0 if s1[r-1] == s2[c-1] else 1)
			cur[c] = min(edit, deletion, insertion)

			# damerau
			if (damerau and r > 1 and c > 1 and s1[r-1] == s2[c-2]
					and s1[r-2] == s2[c-1] and s1[r-1] != s2[c-1]):
				cur[c] = min(cur[c], prevprev[r-2] + 1)

	return 1-float(cur[-1])/max(rows-1, cols-1)

# Copied from Jellyfish (https://pypi.python.org/pypi/jellyfish/0.3.2)
def jaro_winkler(ying, yang, long_tolerance=False, winklerize=True):
	ying_len = len(ying)
	yang_len = len(yang)

	if not ying_len or not yang_len:
		return 0

	min_len = min(ying_len, yang_len)
	search_range = (min_len // 2) - 1
	if search_range < 0:
		search_range = 0

	ying_flags = [False]*ying_len
	yang_flags = [False]*yang_len

	# looking only within search range, count & flag matched pairs
	common_chars = 0
	for i, ying_ch in enumerate(ying):
		low = i - search_range if i > search_range else 0
		hi = i + search_range if i + search_range < yang_len else yang_len - 1
		for j in xrange(low, hi+1):
			if not yang_flags[j] and yang[j] == ying_ch:
				ying_flags[i] = yang_flags[j] = True
				common_chars += 1
				break

	# short circuit if no characters match
	if not common_chars:
		return 0

	# count transpositions
	k = trans_count = 0
	for i, ying_f in enumerate(ying_flags):
		if ying_f:
			for j in xrange(k, yang_len):
				if yang_flags[j]:
					k = j + 1
					break
			if ying[i] != yang[j]:
				trans_count += 1
	trans_count /= 2

	# adjust for similarities in nonmatched characters
	common_chars = float(common_chars)
	weight = ((common_chars/ying_len + common_chars/yang_len +
			  (common_chars-trans_count) / common_chars)) / 3

	# winkler modification: continue to boost if strings are similar
	if winklerize and weight > 0.7:
		# adjust for up to first 4 chars in common
		j = min(min_len, 4)
		i = 0
		while i < j and ying[i] == yang[i] and ying[i]:
			i += 1
		if i:
			weight += i * 0.1 * (1.0 - weight)

		# optionally adjust for long strings
		# after agreeing beginning chars, at least two or more must agree and
		# agreed characters must be > half of remaining characters
		if (long_tolerance and min_len > 4 and common_chars > i+1 and
				2 * common_chars >= min_len + i):
			weight += ((1.0 - weight) * (float(common_chars-i-1) / float(ying_len+yang_len-i*2+2)))

	return weight



def longest_common_substring(s1, s2):
	m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
	longest, x_longest = 0, 0
	for x in xrange(1, 1 + len(s1)):
		for y in xrange(1, 1 + len(s2)):
			if s1[x - 1] == s2[y - 1]:
				m[x][y] = m[x - 1][y - 1] + 1
				if m[x][y] > longest:
					longest = m[x][y]
					x_longest = x
			else:
				m[x][y] = 0
	return float(longest)/max(len(s1), len(s2)) #s1[x_longest - longest: x_longest]
