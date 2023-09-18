from .modules import deepcopy


class neuralnet():
	'''
	Class neuralnet is a Simple NeuralNet.

	Attributes
	----------
	weights: list
	Weights
	
	path: str
	Name of weights file

	network: list
	Structure of NeuralNet

	Methods
	-------
	_clear_deltaweights():
		Sets self.deltaweights values to zero

	_main(net: list) -> list:
		Calculating neuralnet
		Returns output layer

	_teach(inputnet: list, rightanswer: list, koof: float):
		Calculating error and writes delta to self.deltaweights

	teach(inputnet: list, rightanswer: list, koof: float):
		Trains a neural network
		Using _teach to get deltaweights and edit weights with it

	teach_dataset(data: list, koof: float):
		Trains a neural network by dataset
		Using _teach to get deltaweights and edit weights with it

	action(inputnet: list) -> list:
		Calculating and return output layer with _main

	save(path: str):
		Saving self.network and self.weights to file
	'''

	def __init__(self, weights: list = [], path: str = '', network: list = []):
		if ((not network) and (not path)):
			print("Bad input data network:", network, "| path:", path)
		elif network:
			self.network = []
			for i in range(len(network)):
				if (i != len(network) - 1):
					network[i] += 1
				self.network.append(network[i] * [0.0])
		self.weights = []
		if path:
			with open(path, "r") as file:
				self.network = []
				for i in list(map(int, file.readline().split())):
					self.network.append(i * [0.0])
				for s in file:
					List = list(map(float, s.split()))
					if (List):
						self.weights.append(List)
		elif weights:
			self.weights = weights
		elif network:
			for i in range(1, len(network)):
				for g in range(network[i]):
					if ((i != len(network) - 1) and (network[i] - 1 == g)):
						continue
					self.weights.append([])
					for k in range(network[i - 1]):
						self.weights[-1].append(1 / network[i - 1])
		else:
			print("Bad data in NeuralNet init")
		self.deltaweights = deepcopy(self.weights)
		self._clear_deltaweights()
		self._error = []
		for i in range(1, len(self.network)):
			self._error.append([0.0] * (len(self.network[i]) - (1 if (i != len(self.network) - 1) else 0)))
		

	def _clear_deltaweights(self):
		'''Sets self.deltaweights values to zero
		'''
		for x in range(len(self.deltaweights)):
			for y in range(len(self.deltaweights[x])):
				self.deltaweights[x][y] = 0

	def _main(self, net: list) -> list:
		'''Calculating neuralnet
		Returns output layer

		Parameters
		----------
		net : list
			Values of net
		'''
		I = 0
		for j in range(len(net) - 1):
			for y in range(len(net[j + 1])):
				if ((len(net[j + 1]) - 1 == y) and (len(net) - 1 != j + 1)):
					net[j + 1][y] = 1.0
					continue
				x = 0
				for i in range(len(net[j])):
					x += net[j][i] * self.weights[I][i]
				if (x > 1):
					x = 1 + .01 * (x - 1)
				elif (x < 0):
					x *= .01
				net[j + 1][y] = x
				I += 1
		return net

	def _teach(self, inputnet: list, rightanswer: list, koof: float):
		'''Calculating error and writes delta to self.deltaweights

		Parameters
		----------
		inputnet: list
			Values of net
		
		rightanswer: list
			Right output data for inputnet
		
		koof: float
			The value by which to multiply self.deltaweights values
		'''
		self.network = self._main([inputnet + [1.0]] + self.network[1:])
		Error = deepcopy(self._error)
		for i in range(len(self.network[-1])):
			Error[-1][i] = rightanswer[i] - self.network[-1][i]
		a = len(self.weights) - 1
		for x in range(len(Error) - 1, 0, -1):
			for y in range(len(Error[x]) - 1, -1, -1):
				for i in range(len(Error[x - 1]) - 1, -1, -1):
					Error[x - 1][i] += Error[x][y] * self.weights[a][i]
				a -= 1
		i = 0
		for x in range(len(Error)):
			for y in range(len(Error[x])):
				for a in range(len(self.weights[i])):
					self.deltaweights[i][a] += Error[x][y] * koof * self.network[x][a] * (.01 if (
						not 0 <= self.network[x + 1][y] <= 1) else 1)
				i += 1

	def teach(self, inputnet: list, rightanswer: list, koof: float):
		'''Trains a neural network
		Using _teach to get deltaweights and edit weights with it

		Parameters
		----------
		inputnet: list
			Values of net
		
		rightanswer: list
			Right output data for inputnet
		
		koof: float
			The value by which to multiply self.deltaweights values
		'''
		self._clear_deltaweights()
		self._teach(inputnet, rightanswer, koof)
		for x in range(len(self.deltaweights)):
			for y in range(len(self.deltaweights[x])):
				self.weights[x][y] += self.deltaweights[x][y]

	def teach_dataset(self, data: list, koof: float):
		'''Trains a neural network by dataset
		Using _teach to get deltaweights and edit weights with it

		Parameters
		----------
		data: list
			Dataset with structure like [[inputnet, rightanswer], ...]
		
		koof: float
			The value by which to multiply self.deltaweights values
		'''
		self._clear_deltaweights()
		for i in range(len(data)):
			self._teach(data[i][0], data[i][1], 1 / len(data))
		for x in range(len(self.deltaweights)):
			for y in range(len(self.deltaweights[x])):
				self.weights[x][y] += self.deltaweights[x][y] * koof

	def action(self, inputnet: list) -> list:
		'''Calculating and return output layer with _main

		Parameters
		----------
		inputnet: list
			Input values of net
		'''
		return self._main([inputnet + [1.0]] + self.network[1:])[-1].copy()

	def save(self, path: str = "weights.data"):
		'''Saving self.network and self.weights to file

		Parameters
		----------
		path: str
			Path to file
		'''
		with open(path, "w") as file:
			for x in self.network:
				file.write(str(len(x)) + " ")
			file.write("\n")
			for x in self.weights:
				file.write(" ".join(map(str, x)) + "\n")
