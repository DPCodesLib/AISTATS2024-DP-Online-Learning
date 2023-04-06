#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Algorithm7(T,M,N,d,U,Gvt,epi=None):
    if epi==None:
        epi = 1/np.sqrt(T)
    print(epi)
    ita = np.sqrt(d/T)
    err = 1e-10

    X = np.ones((T,M,N))/N
    Lt = np.zeros((T,N))
    Rtj = np.zeros((T,M))
    Rt = np.zeros(T)
    #Zt = np.zeros((M,N))
    gtj = np.zeros((M,T,N))

    #np.random.seed(202303)
    MU = MVEE(U)
    Zt = ARSampling(MU,epi,d,M*T)
    Zt = Zt.reshape(T,M,d)

    for t in range(T):
        vt = Gvt[t]
        Lt[t] = calLt(U,vt)
        #print(Lt[t])

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
                Rtj[t][j] = sumXL - min(sum(Lt[:t+1]))
                #print(min(sum(Lt[0:t+1])))
        #print(Rtj[t])
        Rt[t] = sum(Rtj[t])/M

        H = np.eye(N)+np.dot(np.dot(U,MU),U.T)
        if t<T-1:
            for iz in range(M):
                gtj[iz][t] = Lt[t]+np.dot(U,Zt[t][iz])
                #print(gtj)
                #print(X[iz])
            #for iz in range(M):
                X[t+1][iz] = _argmin_7(gtj[iz],N,ita,H)
                #print(X[t+1][iz])
        else:
            break
        #print('\n')
    return Rt

