from threading import Thread
from random import random
from time import perf_counter


def deepcopy(data: list):
    output = []
    for i in data:
        output.append(i.copy())
    return output
