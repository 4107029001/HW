# -*- coding: utf-8 -*-
"""
Created on Fri May 14 21:25:57 2021

@author: carll
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:41:01 2021

@author: carll
"""
from itertools import chain, combinations
from sys import maxsize
import numpy as np
import time
import matplotlib.pyplot as plt

'''DP'''
D = {}
P = {}

def travel(n,W): 
    V = []
    for i in range(n):
        V.append(i+1)
#    V = str(V)
    del V[0]
#    V = "".join()
#    print('V:',V)
    
    subsets = chain.from_iterable(combinations(V,r)for r in range(len(V)+1))
#   print(subsets)
    subsets = list(subsets)
    
#    print('subsets without v1:',subsets)    
#    print('subsets without v1:',str(subsets)) 
    for i in range(2,n+1):
        D[i,subsets[0]] = W[i-1][0]
#        D[subsets[0],i] = W[0][i-1]
#    print('Dict:',D,'\n')
#    D_copy = {key[0]: value for key, value in D.items()}
#    print('D_copy:',D_copy)
    for k in range(1,n-1):
#        subsets = chain.from_iterable(combinations(V,k))
        subsets = chain.from_iterable(combinations(V,r) for r in range(k,k+1))
        subsets = list(subsets)
#        print('subsets with {} vertices:'.format(k),subsets,'\n') 
        A = subsets
#        print('A:',A,'\n')
        for a in range(len(A)):
#            A[a] = set([A[a]])
            A[a] = set(A[a])
            V = set(V)
            A_withoutj = V - A[a]
            A_withoutj = sorted(A_withoutj)
            A[a] = sorted(A[a])
            A[a] = tuple(A[a])
            A_withoutj = tuple(A_withoutj)


#            MIN = []

            for i in range(len(A_withoutj)):
#                print('i:',A_withoutj[i])
                current = maxsize
                currentj = 0
                for j in range(len(A[a])):
#                    print(A[a][j])
                    if len(A[a])>=2: 
#                        print('A[a]',A[a])

                        E = []
                        E.append(A[a][j])                                                                        
                        E = set(E)
                        A[a] = set(A[a])
                        B = A[a] - E
                        A[a] = sorted(A[a])
                        A[a] = tuple(A[a])
                        E = tuple(E)
                        B = sorted(B)
                        B = tuple(B)
#                        print('A[a]',A[a])
#                        print('E:',E)
#                        print('B',B)
#                        print('j:',A[a][j])
                        
#                        D[A_withoutj[i],A[a]] =  W[i-2][A[a][j]-1] + D[A[a][j],B]
                        temp = W[A_withoutj[i]-1][A[a][j]-1] + D[A[a][j],B]
#                        MIN.append(W[i-2][A[a][j]-1] + D[A[a][j],B])
                        if temp<current:
                            current = temp
                            currentj = A[a][j]
                        
#                        print('比大小',MIN)
                        D[A_withoutj[i],A[a]] = current
                        P[A_withoutj[i],A[a]] = currentj
                        
                    elif len(A[a]) == 1:
                        D[A_withoutj[i],A[a]] =  W[A_withoutj[i]-1][A[a][j]-1] + D[A[a][j],()]
                        
#                        print('W[i-1][A[a][j]-1]',W[i-2][A[a][j]-1])
#                        print('A[a][j]',A[a][j])
#                    print(D)
#    print(D)
#    MIN2 = []
    V2 = []
    for i in range(n):
        V2.append(i+1)
    del V2[0]
    current = maxsize
    currentj = 0
#    print(V2)
    for j in range(2,n+1):
        V_copy = V.copy()
        V_copy.remove(j)
        V_copy = sorted(V_copy)
        V_copy = tuple(V_copy)
        temp= W[0][j-1] + D[j,V_copy]
        
        V_copy = tuple(V_copy)
#        print('V_copy',V_copy)
#        MIN2.append(W[0][j-1] + D[j,V])
#        print('temp',temp)
        
        if temp<current:
            
            current = temp
            currentj = j

#            print('current',current)
#        print('比大小',MIN2)
        V2 = tuple(V2)
        D[1,V2] = current
        P[1,V2] = currentj
        V = set(V)
#        print('current:',current)
#    print('minj:',P[1,V2])
    minlength = D[1,V2] 
    V = tuple(V)
      
#    print(D)
#    print('DPminlength',minlength)                
    return minlength


'''2-opt'''
def cost_change(matrix, n1, n2, n3, n4):
    return matrix[n1][n3] + matrix[n2][n4] - matrix[n1][n2] - matrix[n3][n4]

def two_opt(route, matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(matrix, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best

def cost(matrix, route):
   return matrix[np.roll(route, 1), route].sum()         

 

           
Data1 = [] #n
Data2 = [] #DPt
Data3 = [] #2-opt time
Data4 = [] #平均誤差
for i in range(4,21):    
    n=i
    
    all_vertice = list(range(n))
    
    W = np.random.randint(low = 1, high = 30, size=(n, n))
    for i in range(n):
        for j in range(n):
            if i==j:
                W[i][j] = 0
            elif i>j:
                W[i][j] = W[j][i]
    print('n=',n)
    
    '''計算DPtime'''
    dpstart = time.time()
    length = travel(i,W)
    dpend = time.time() 
    DPt = dpend-dpstart
#    plt.plot(n,t)
    print('DPminlength',length)  
    print('DPtime:',DPt,'\n')

    opt_start = time.time() 
    best_route = two_opt(all_vertice, W) 
    
    print('2-opt-mincost:',cost(W, best_route))
    opt_end = time.time()
    optt = opt_end - opt_start
    print('2opt_time:',optt)
    print(best_route)
    #計算相對誤差
    error = abs((cost(W, best_route) - length))/length
    print('相對誤差',error,'\n')

    

    
#   畫圖
    Data1.append(n)
    Data2.append(DPt)
    Data3.append(optt)
    Data4.append(error)
#    Data2 = plt.plot(n, backtracking_time,'g',label='backtracking')
#print(W)
plt.plot(Data1,Data2,'r',Data1,Data3,'g')
plt.title('Time')
plt.xlabel('n')
plt.ylabel('time')
plt.show()

plt.plot(Data1,Data4)
plt.title('Efficiency')
plt.xlabel('n')
plt.ylabel('error')
plt.show()

 


 