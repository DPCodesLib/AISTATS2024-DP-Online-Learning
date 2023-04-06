#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Algorithm6(T,M,N,d,U,Gvt,epi = None):
    if epi==None:
        epi = 1/np.sqrt(T)
    lamb = 2/epi
    ita = np.sqrt(np.log(N)/T)
    err = 1e-10

    X = np.ones((T,M,N))/N
    Lt = np.zeros((T,N))
    Rtj = np.zeros((T,M))
    Rt = np.zeros(T)
    #Zt = np.zeros((M,N))
    gtj = np.zeros((M,T,N))

    #np.random.seed(20230301)

    for t in range(T):
        #print('X=',X[t])

        vt = Gvt[t]
        Lt[t] = calLt(U,vt)
        #print('Lt=',Lt[t])

        for j in range(M):
            sumXL = 0
            for k in range(t+1):
                #print(X[k][j])
                sumXL += np.dot(X[k][j],Lt[k].reshape(1,N).T)
                #print(sumXL)
            if t==0:
                Rtj[t][j] = sumXL - min(Lt[t])
                #print(min(Lt[t]))
            else:
                Rtj[t][j] = sumXL - min(sum(Lt[0:t+1]))
                #print(min(sum(Lt[0:t+1])))
        #print('Rtj=',Rtj[t])
        Rt[t] = sum(Rtj[t])/M
        #print('Rt=',Rt[t])
        if t<T-1:
            for iz in range(M):
                Zt = np.random.laplace(0,lamb,N)
                gtj[iz][t] = Lt[t]+Zt
                #print(gtj)
                #print(X[iz])
            #for iz in range(M):
                X[t+1][iz] = _argmin(gtj[iz],N,ita)
                #print('X=',X[t+1][iz])
                #print(t,iz)
                #print('\n')
        else:
            break
        #print('\n')
    return Rt
    

