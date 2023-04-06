#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Algorithm9(T,M,N,d,U,Gvt,epi = None):
    if epi==None:
        epi = 1/np.sqrt(T)
    lamb = np.log2(T)/epi
    ita = np.sqrt(np.log(N)/T)
    err = 1e-10

    X = np.zeros((T,M,N))/N
    X[0] = np.ones((M,N))/N
    Lt = np.zeros((T,N))
    Rtj = np.zeros((T,M))
    Rt = np.zeros(T)
    n_len = int(np.log2(T))
    nLap = np.random.laplace(0,lamb,size=(n_len,N))
    V = np.zeros((M,T,N))

    #np.random.seed(202303019)

    for t in range(T):
        #print(X[t])

        vt = Gvt[t]
        Lt[t] = calLt(U,vt)
        #print('Lt = ',Lt[t])

        for j in range(M):
            sumXL = 0
            for k in range(t+1):
                #print(X[k][j])
                sumXL += np.dot(X[k][j],Lt[k].reshape(1,N).T)
                #print('sumXL = ',sumXL)
            if t==0:
                Rtj[t][j] = sumXL - min(Lt[t])
                #print(min(Lt[t]))
            else:
                Rtj[t][j] = sumXL - min(sum(Lt[:t+1]))
                #print('min of Lt = ',min(sum(Lt[:t+1])))
        #print(Rtj[t])
        Rt[t] = sum(Rtj[t])/M
        if t<T-1:
            TLt = sum(Lt[:t+1])
            st = calSt(t)
            for iz in range(M):
                epsi_t = np.random.laplace(0,lamb,size=(st,N))
                V[iz][t] = sum(epsi_t) + TLt + sum(nLap[:n_len-st])
                #print(gtj)
                #print(X[iz])
            #for iz in range(M):
                X[t+1][iz] = _argmin(V[iz],N,ita)
                #print(X[t+1][iz])
        else:
            break
        #print('\n')
    return Rt

def TreeAgg(lt,T,epsi):
    n_len = np.log2(T)
    lambd = n_len/epsi
    n = np.random.laplace(0,lambd,size=n_len)
    V = np.zeros(T)
    for t in range(T):
        Lt = sum(lt[:t])
        st = calSt(t)
        epsi_t = np.random.laplace(0,lambd,size=st)
        V[t] = sum(epsi_t) + Lt + sum(n[:n_len-st])
    return V

