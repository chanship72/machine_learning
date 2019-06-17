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

	sdict = dict()
	wordList = dict()
	wordTagList = dict()
	firstTagList = dict()
	tagList = dict()

	for sentence in train_data:
		if tags.index(sentence.tags[0]) not in firstTagList:
			firstTagList[tags.index(sentence.tags[0])] = 1
		else:
			firstTagList[tags.index(sentence.tags[0])] += 1
		for tag, word in zip(sentence.tags,sentence.words):
			if tag not in wordTagList:
				wordTagList[tag] = dict()
				if word not in wordTagList[tag]:
					wordTagList[tag][word] = 1
				else:
					wordTagList[tag][word] += 1
			else:
				if word not in wordTagList[tag]:
					wordTagList[tag][word] = 1
				else:
					wordTagList[tag][word] += 1

			if word not in wordList:
				wordList[word] = 1
			else:
				wordList[word] += 1
			if tag not in tagList:
				tagList[tag] = 1
			else:
				tagList[tag] += 1
			if tag not in sdict:
				sdict[tag] = [word]
			else:
				sdict[tag].append(word)

	vsum = sum(firstTagList.values())
	pi = [v/vsum for k,v in sorted(firstTagList.items())]

	# vsum = sum(tagList.values())
	# pi = [v/vsum for k,v in sorted(tagList.items())]

	A = np.zeros((len(tags), len(tags)))
	for state in tags:
		for sentence in train_data:
			for tag1, tag2 in zip(sentence.tags[0::2], sentence.tags[1::2]):
				if state == tag1:
					A[tags.index(state)][tags.index(tag2)] += 1

	for k in range(len(A)):
		if np.sum(A[k,:])>0:
			A[k,:] = A[k,:]/np.sum(A[k,:])

	B = np.zeros((len(tags), len(wordList.keys())))
	words = [*wordList]
	for k,v in wordTagList.items():
		for word in v:
			B[tags.index(k)][words.index(word)] = wordTagList[k][word]
		# B[tags.index(k), :] = B[tags.index(k), :] / len(v)

	for k in range(len(B)):
		if np.sum(B[k,:])>0:
			B[k,:] = B[k,:]/np.sum(B[k,:])

	dState = {tag:i for i,tag in enumerate(tags)}
	dOseq = {w:j for j,w in enumerate(words)}

	model = HMM(pi, A, B, dOseq, dState)

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
		mtag = model.viterbi(np.array(sentence.words))
		tagging.append(mtag)

	return tagging

