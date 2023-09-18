import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from neuralnet import neuralnet
from neuralnet import tools

Net = neuralnet(network = [2, 10, 1])

data = [
	[[0, 0], [0]],
	[[0, 1], [.5]],
	[[1, 0], [.5]],
	[[1, 1], [1]],]

output = []
for i in range(len(data)):
	output.append(Net.action(data[i][0]))
print(output)

i = 0
while((tools.get_err(Net, data) > .01) and (i < 10**4)):
	Net.teach_dataset(data, .1)
	i+=1

tools.dataset_save("dataset.data", data)
print(tools.dataset_read("dataset.data"))

output = []
for i in range(len(data)):
	output.append(Net.action(data[i][0]))
print(output)

