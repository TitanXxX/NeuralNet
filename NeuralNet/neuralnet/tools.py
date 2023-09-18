
def dataset_save(path: str, data: list):
	'''Saves dataset to file

	Parameters
	----------
	path: str
		Path to file
	
	data: list
		Dataset with structure like [[inputnet, rightanswer], ...]
	'''
	with open(path, "w") as f:
		for i in data:
			f.write(" ".join(map(str, i[0])) + "\n" + " ".join(map(str, i[1])) + "\n")


def dataset_read(path: str) -> list:
	'''Reads and returns dataset(with structure like [[inputnet, rightanswer], ...]) from file
	
	Parameters
	----------
	path: str
		Path to file
	'''
	data = []
	with open(path, "r") as f:
		flag = True
		for line in f:
			if (flag):
				data.append([list(map(float, line.split()))])
			else:
				data[-1].append(list(map(float, line.split())))
			flag = not flag
	return data


def get_err(net, data: list) -> float:
	'''Calculates error on dataset
	
	Parameters
	----------
	net: neuralnet
		NeuralNet

	data: list
		Dataset with structure like [[inputnet, rightanswer], ...]
	'''
	out = 0
	for i in data:
		result = net.action(inputnet = i[0])
		for g in range(len(result)):
			out += abs(result[g] - i[1][g])
	return out / len(data)



