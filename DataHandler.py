import pickle
import numpy as np
from scipy.sparse import csr_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import random

def transpose(mat):
	coomat = sp.coo_matrix(mat)
	return csr_matrix(coomat.transpose())

def negSamp(temLabel, sampSize, nodeNum):
	negset = [None] * sampSize
	cur = 0
	while cur < sampSize:
		rdmItm = np.random.choice(nodeNum)
		if temLabel[rdmItm] == 0:
			negset[cur] = rdmItm
			cur += 1
	return negset

def transToLsts(mat, mask=False, norm=False):
	shape = [mat.shape[0], mat.shape[1]]
	coomat = sp.coo_matrix(mat)
	indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
	data = coomat.data.astype(np.float32)

	if norm:
		rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
		colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
		for i in range(len(data)):
			row = indices[i, 0]
			col = indices[i, 1]
			data[i] = data[i] * rowD[row] * colD[col]

	# half mask
	if mask:
		spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
		data = data * spMask

	if indices.shape[0] == 0:
		indices = np.array([[0, 0]], dtype=np.int32)
		data = np.array([0.0], np.float32)
	return indices, data, shape

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = 'Data/yelp/'
		elif args.data == 'tmall':
			predir = 'Data/tmall/'
		elif args.data == 'gowalla':
			predir = 'Data/gowalla/'
		elif args.data == 'ml10m':
			predir = 'E:/codes/MODEL/HCCF/Data/ml10m/'
		elif args.data == 'amazon':
			predir = 'E:/codes/MODEL/HCCF/Data/amazon/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		if args.data == 'ciao':
			self.predir = 'E:/codes/datasets/REC/Ciao/'
		elif args.data == 'epinions':
			self.predir = 'E:/codes/datasets/REC/Epinions/'



	def add_noice_experiment(self):
		return

	def LoadData(self):
		with open(self.trnfile, 'rb') as fs:
			trnMat = (pickle.load(fs) != 0).astype(np.float32)
		# test set
		with open(self.tstfile, 'rb') as fs:
			tstMat = pickle.load(fs)
		tstLocs = [None] * tstMat.shape[0]
		tstUsrs = set()
		for i in range(len(tstMat.data)):
			row = tstMat.row[i]
			col = tstMat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))

		self.trnMat = trnMat
		self.tstLocs = tstLocs
		self.tstUsrs = tstUsrs
		args.edgeNum = len(trnMat.data)
		args.user, args.item = self.trnMat.shape



class DataHandler_ciaoAndEpinions:
	def __init__(self):
		if args.data == 'ciao':
			self.predir = 'E:/codes/datasets/REC/Ciao/'
		elif args.data == 'epinions':
			self.predir = 'E:/codes/datasets/REC/Epinions/'

	def add_noice_experiment(self):
		return

	def LoadData(self, test_prop):

		if args.data == 'epinions':
			click_f = np.loadtxt(self.predir + 'ratings_data.txt', dtype=np.int32)
		else:
			click_f = np.loadtxt(self.predir + 'rating_with_timestamp.txt', dtype=np.int32)


		self.user_count = 0
		self.item_count = 0
		pos_list = []

		for s in click_f:
			uid = s[0]
			iid = s[1]

			if uid > self.user_count:
				self.user_count = uid
			if iid > self.item_count:
				self.item_count = iid

			pos_list.append([uid, iid])
			# self.train_users.append(uid)
			# self.train_items.append(iid)
		random.shuffle(pos_list)

		random.shuffle(pos_list)
		num_test = int(len(pos_list) * test_prop)
		self.test_set = pos_list[:num_test]
		self.valid_set = pos_list[num_test:2 * num_test]
		self.train_set = pos_list[2 * num_test:]

		self.train_users = []
		self.train_items = []
		for interection in self.train_set:
			self.train_users.append(interection[0])
			self.train_items.append(interection[1])

		self.trnMat = csr_matrix((np.ones(len(self.train_users)), (self.train_users, self.train_items)), shape=(self.user_count + 1, self.item_count + 1))

		self.tstUsrs = []
		self.tstLocs = [None] * self.trnMat.shape[0]
		for i, interection in enumerate(self.test_set) :
			curr_user = interection[0]
			curr_item = interection[1]
			self.tstUsrs.append(curr_user)
			if self.tstLocs[curr_user] is None:
				self.tstLocs[curr_user] = [curr_item]
			else:
				self.tstLocs[curr_user].append(curr_item)
		self.tstUsrs = np.array(list(self.tstUsrs))
		args.edgeNum = len(self.trnMat.data)
		args.user, args.item = self.trnMat.shape
		pass


