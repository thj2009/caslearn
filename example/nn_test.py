#Read data from csv file
import pandas as pd
import numpy as np
import casadi as cas
from caslearn import NN


df = pd.read_csv('HighThData.csv')

M1 = df.M1.values.tolist()
M2 = df.M2.values.tolist()
M3 = df.M3.values.tolist()
Sup = df['Support '].values.tolist()
#Relative atomic percentage of Mx in M1+M2+M3
M1_mp = df['M1_mol%'].values.tolist()
M2_mp = df['M2_mol%'].values.tolist()
M3_mp = df['M3_mol%'].values.tolist()
#Temperature (C)
T = df['Temp']
#total flow in (mL/min)
F = df['Total_flow'].values.tolist()
#each composition flow in (mL/min)
Ar_F = df['Ar_flow'].values.tolist()
CH4_F = df['CH4_flow'].values.tolist()
O2_F = df['O2_flow'].values.tolist()
#contact time(s)
CT = df['CT'].values.tolist()

#Output
xCH4 = df['CH4_conv'].values.tolist()
yC2H6 = df['C2H6y'].values.tolist()
yC2H4 = df['C2H4y'].values.tolist()
yCO = df['COy'].values.tolist()
yCO2 = df['CO2y'].values.tolist()


#Convert Sup to number
#print(np.unique(Sup))
#['Al2O3' 'BEA' 'BN' 'CeO2' 'MgO' 'Nb2O5' 'SiC' 'SiCnf' 'SiO2' 'TiO2' 'ZSM-5' 'ZrO2' 'n.a.']

#1:Al2O3;2:BEA; 3：BN;4：CeO2;5:MgO;6:Nb2O5;7:SiC;8:SiCnf;9:SiO2;10:TiO2
#11:ZSM-5;12:ZrO2;13:n.a.;

for n,i in enumerate(Sup) :
    Sup[n] = 1 if i =='Al2O3' else 2 if i =='BEA' else 3 if i =='BN' \
    else 4 if i =='CeO2' else 5 if i =='MgO' else 6 if i =='Nb2O5'\
    else 7 if i =='SiC' else 8 if i =='SiCnf' else 9 if i =='SiO2'\
    else 10 if i =='TiO2' else 11 if i =='ZSM-5' else 12 if i =='ZrO2' else 13

#Conver M1,M2,M3 in to integer

#M1
# print(np.unique(M1))
#1:'Ce' 2:'Co' 3:'Cu' 4:'Eu' 5:'Fe' 6:'Hf' 7:'La' 8:'Mn' 9:'Mo' 
#10:'Nd' 11:'Ni' 12:'Pd' 13:'Tb' 14:'Ti' 15:'V'
#16:'Y' 17:'Zn' 18:'Zr' 19:'n.a.'
for n,i in enumerate(M1) :
    M1[n] = 1 if i =='Ce' else 2 if i =='Co' else 3 if i =='Cu' \
    else 4 if i =='Eu' else 5 if i =='Fe' else 6 if i =='Hf'\
    else 7 if i =='La' else 8 if i =='Mn' else 9 if i =='Mo'\
    else 10 if i =='Nd' else 11 if i =='Ni' else 12 if i =='Pd'\
    else 13 if i =='Tb' else 14 if i =='Ti' else 15 if i =='V' \
    else 16 if i =='Y' else 17 if i =='Zn' else 18 if i =='Zr' else 19

#M2
#print(np.unique(M2))
#1:'Ba' 2:'Ca' 3:'Fe' 4:'K' 5:'Li' 6:'Mg' 7:'Na' 8:'Sr' 9:'Zn' 10:'n.a.'
for n,i in enumerate(M2) :
    M2[n] = 1 if i =='Ba' else 2 if i =='Ca' else 3 if i =='Fe' \
    else 4 if i =='K' else 5 if i =='Li' else 6 if i =='Mg'\
    else 7 if i =='Na' else 8 if i =='Sr' else 9 if i =='Zn' else 10
#M3
#print(np.unique(M3))
#1:'Mo' 2:'W' 3:'n.a.'
for n,i in enumerate(M3) :
    M3[n] = 1 if i =='Mo' else 2 if i =='W' else 3 
    
print(1)
    
 #neural network

X= np.column_stack((M1,M2,M3,Sup,M1_mp,M2_mp,M3_mp,T,F,Ar_F,CH4_F,CT))
#Shuffle x 
from sklearn.utils import shuffle
X,xCH4,yC2H6,yC2H4,yCO,yCO2 = shuffle(X,xCH4,yC2H6,yC2H4,yCO,yCO2,random_state=0)
#t = np.round(len(x)*0.8)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X)


n = 7000
xs = [X[:n, :], X[:n, :], X[:n :], X[:n :], X[:n, :]]
ys = [xCH4[:n], yC2H6[:n], yC2H4[:n], yCO[:n], yCO2[:n]]

# xs = [X[:100, :]]
# ys= [xCH4[:100]]
# define the neural network structure
# nin: number of input
# nout: number of output
# nhidden: number of hidden layer
# nhDList: List of dimension of the hidden layer

print(1)
#number of hidden layer nL
nL = 5
#number of nodes for each layer nD
nD = 3
#List of dimension of hidden layer
nh = [nD] * nL
nn = NN(nin=12, nout=5, nhidden=nL, nhDList=nh)
nn.fit(xs, ys)


from sklearn.metrics import mean_squared_error
#Predict yC2H6
ypred = nn.predict(X[12000:, :], nout=0)


import matplotlib.pyplot as plt

fig = plt.figure(dpi=200, figsize=(2, 2))
ax = fig.gca()
ax.set_aspect('equal')
ax.plot(xCH4[12000:], ypred, 'ro')
# ax.plot([-1, 2], [-1, 2], 'k:')
# ax.set_xlim([-1, 2])
# ax.set_ylim([-1, 2])
ax.tick_params(which='both', direction='in', labelsize=6)
plt.show()

