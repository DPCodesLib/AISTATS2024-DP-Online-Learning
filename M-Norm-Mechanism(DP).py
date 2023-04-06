#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Algorithm11(T,M,N,d,U,Gvt,epi=None):
    if epi==None:
        epi = 1/np.sqrt(T)
    ita = np.sqrt(d/T)
    err = 1e-10

    X = np.zeros((T,M,N))
    X[0] = np.ones((M,N))/N
    Lt = np.zeros((T,N))
    Rtj = np.zeros((T,M))
    Rt = np.zeros(T)

    #np.random.seed(20230321)
    V = np.zeros((M,T,d))
    MU = MVEE(U)
    nLap = ARSampling_9(MU,epi,d,M*T,T)
    nLap = nLap.reshape(T,M,d)
    n_len = int(np.log2(T))
    epsi_t = ARSampling_9(MU,epi,d,n_len*M,T)
    epsi_t = epsi_t.reshape(M,n_len,d)

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
            TVt = sum(vt[:t+1])
            st = calSt(t)
            for iz in range(M):    
                V[iz][t] = sum(epsi_t[iz][:st]) + TVt + sum(nLap[t][:n_len-st])
                #print(gtj)
                #print(X[iz])
            #for iz in range(M):
                X[t+1][iz] = _argmin_11(V[iz],N,ita,H,U)
                #print(X[t+1][iz])
        else:
            break
        #print('\n')
    return Rt

