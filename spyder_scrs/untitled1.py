# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:21:55 2017

@author: atpandey
"""
#%%
dict_a={
1:'a',
2:'b',
3:'c',
4:'d',
5:'e',
6:'f',
7:'g',
8:'h',
9:'i',
10:'j',
11:'k',
12:'l',
13:'m',
14:'n',
15:'o',
16:'p',
17:'q',
18:'r',
19:'s',
20:'t',
21:'u',
22:'v',
23:'w',
24:'x',
25:'y',
26:'z'      
        }

keys=list(dict_a.keys())
vals=list(dict_a.values())

    
print("keys",keys)
print("values",vals)


inp_w_n=input("encode or decode (1: encode, 2: decode)?")
if inp_w_n == '1':
    inp=input("enter your word here:")

    for i in inp.lower():
        #print(i)
        if i in vals:
            pos=vals.index(i)
            print(keys[pos],end='_')

if inp_w_n == '2':
    inp=input("enter your numbers here:")
    nums=inp.split('_')
    for i in nums:
        #print(i)
        try:
            if int(i) in keys:
                pos=keys.index(int(i))
                print(vals[pos],end='')
        except:
            pass
        
#8_5_12_12_15
             
    