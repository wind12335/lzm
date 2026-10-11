import pickle

with open("./nmt_4layers_baseline.pkl", "rb") as file:
    res = pickle.load(file)

for key in zip(res):
    print(key)