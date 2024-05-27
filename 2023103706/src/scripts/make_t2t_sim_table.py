import os
import numpy as np
from tqdm import tqdm

data_path = 'dataset/zsl/CUB/CUB_200_2011/attributes/image_attribute_labels.txt'

data = np.full((11788, 312), False, dtype=bool)

with open(data_path, 'r') as data_file:
    lines = data_file.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        if line[2]==str(1):
            data[int(line[0])-1][int(line[1])-1] = True

def get_t2t_sim(test):
    best_score = -1
    best_id = 0
    with open('dataset/cub/train.txt', 'r') as trf:
        trains = trf.readlines()
        for train in tqdm(trains):
            train = int(train.strip())
            score = 0
            for i in range(312):
                if data[test-1][i] == data[train-1][i]:
                    score += 1
            if score > best_score:
                best_score = score
                best_id = train
    return best_id


def visual_test():
    num = 0
    with open('dataset/cub/test.txt', 'r') as tef:
        tests = tef.readlines()
        best = []
        with open('dataset/cub/train.txt', 'r') as trf:
            trains = trf.readlines()
            for test in tqdm(tests):
                num+=1
                if num>10: break
                test = int(test.strip())
                best_score = -1
                best_id = 0
                for train in tqdm(trains):
                    train = int(train.strip())
                    score = 0
                    for i in range(312):
                        if data[test-1][i] == data[train-1][i]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_id = train
                best.append([str(best_id)])
        print(best)
        '''
        with open('best.txt', 'w') as bf:
            bf.writelines(best)
