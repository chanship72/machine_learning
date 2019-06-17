import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	###################################################
	# for i in train_data:
	# 	print("train_data:" + str(i.words))
	# print("tags:" + str(tags))

	sdict = dict()
	for sentence in train_data:
		for tag, word in zip(sentence.tags,sentence.words):
			if word not in sdict:
				sdict[word] = [tag]
			else:
				sdict[word].append(tag)

	# print(sdict)
	# print(len(sdict))

	num_dict = dict()
	word_sum = 0
	S = []
	pi = []
	B_dict = dict()
	for key,value in sdict.items():
		S.append(key)
		num_dict[key] = len(value)
		word_sum += len(value)

		kind_prob = dict()
		for term in value:
			if term in kind_prob:
				kind_prob[term] += 1/len(value)
			else:
				kind_prob[term] = 1/len(value)
		B_dict[key] = kind_prob

	# print(B_dict)
	# print(num_dict)
	# print(word_sum)
	# print(S)

	for word in S:
		pi.append(num_dict[word]/word_sum)

	# print(len(pi))
	assert len(pi) == len(S)

	A = np.zeros((len(S), len(S)))

	for keyword in S:
		for sentence in train_data:
			for word1, word2 in zip(sentence.words[0::2], sentence.words[1::2]):
				if keyword == word1:
					A[S.index(keyword)][S.index(word2)] += 1
		A[S.index(keyword)] = A[S.index(keyword)] / np.sum(A[S.index(keyword)])
	# print(A[122])
	# print(A.shape)

	B = np.zeros((len(S),len(tags)))

	for key,value in B_dict.items():
		for k,v in value.items():
			B[S.index(key)][tags.index(k)] = v

	# print(B)
	dtags = dict()
	i = 0
	for tag in tags:
		dtags[tag] = i
		i+=1
	dtags['X'] = i
	dS = dict()
	j = 0
	for s in S:
		dS[s] = j
		j+=1

	print(dtags)
	print(dS)

	model = HMM(pi, A, B, dtags, dS)

	return model


# TODO:
def speech_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	###################################################
	for sentence in test_data:
		print("sentence.words:" + str(sentence.tags))
		mtag = model.viterbi(np.array(sentence.words))
		print("mtag" + str(mtag))
		tagging.append(mtag)

	# print("tagging:" + str(tagging))

	return tagging

