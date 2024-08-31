import scipy as sp

# import networkx as nx
# from matplotlib import pyplot
# max_iter=100 #number of iterations


class mapnmf:
    def __init__(self,alp=1,A,W,X,data,init=1,bet=0.5,max_iter=100):
        node_size = S.shape[0]
        att_size = X.shape[1]
        #H is a matrix between topic1 and topic2
        H = np.random.random((k1,k2))
        #create initialized matrices
        U,V = VU_init.VU_init(X,k1,k2,init,data)
        # W = np.abs(S_ori-1)
        #W_ = 1-W
        # general params
        self.S = S
        self.W = W
        self.W_ = W_
        self.X = X
        self.max_iter = max_iter
        self.U = U
        self.H = H
        self.V = V
        self.alp = alp
        self.bet = bet
    def fit_predict(self):
        def update_U(A,X,alp,U,H,V,bet):
            UUT = U.dot(U.transpose())
            U = U*(4*A.dot(U)+(2.alp*X.dot(H) + 2.bet.Z.dot(V))/(4*(UUT*U).dot(U)+(alp.2.0*)*(HHT*).dot(U)+(alp*2*UH.dot(H.transpose())+bet*2*U.dot(V.transpose().V))
            #V = V*((a*2.0)*S.dot(V)+(lam*A.dot(U) * fdVT).dot(T.transpose()))/(((2.0*a)*(VVT*S_ori)+(2.0*(1.0-a))*(VVT*S_)).dot(V)+(lam*fVT.dot(U.transpose().dot(U)) * fdVT).dot(T.transpose()))
            return U
        
        def update_V(Z,U,H,V):
             V= V*(Z.transpose().dot(U))/(V.(dot(H).transpose()).dot(U))
             return V
        def update_H(A,X,U,H,V):
             H = H*(U.transpose().dot(fdUH*(X.dot(V))))/(U.transpose().dot((fdUH*fUH).dot(V.transpose().dot(V))))
            return H
        def removing_nan(mat):
            nan_list = np.argwhere(np.isnan(mat))
            for i in nan_list:
                mat[i[0],i[1]]=sys.float_info.epsilon
            return mat
        
        start = time.time() #memo start time
        #learning step
        count = 0
        while 1:
            count += 1
            # print loss_function(S,V,U,Z,A,T,lam)
            if self.alp == 0.5:
                self.U = removing_nan(update_U_woPU(self.S,self.X,self.lam,self.U,self.H,self.V))
            else:
                self.U = removing_nan(update_U(self.S,self.W,self.W_,self.X,self.lam,self.U,self.H,self.V,self.rho))
            self.V = removing_nan(update_V(self.S,self.X,self.U,self.H,self.V))
            self.H = removing_nan(update_H(self.S,self.X,self.U,self.H,self.V))
            if count>=self.max_iter:
                break
        elapsed_time = time.time() - start  #measure elapsed time
        print (("optimizing_time:{0}".format(elapsed_time)) + "[sec]")
        return self.U
