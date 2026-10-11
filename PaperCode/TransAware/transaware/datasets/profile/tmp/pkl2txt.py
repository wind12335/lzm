
#!/usr/bin/env python

import pickle

path = 'cost_xyk.pkl'

f = open(path,'rb')
data = pickle.load(f)



with open('./cost_xyk.txt','w') as f:
    print(data,file = f)



