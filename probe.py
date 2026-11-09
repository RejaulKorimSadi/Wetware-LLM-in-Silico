import numpy as np, pilot as P
def caps(sub, K=6):
    r=np.random.default_rng(5); u=r.uniform(-1,1,2200); X=sub.run(u[:,None])[200:]; u=u[200:]
    Xtr,Xte=P.zs(X[:1600],X[1600:]); res=[]
    for k in range(0,K):
        y=np.roll(u,k); W=P.ridge(Xtr,y[:1600,None],1e-2); res.append(round(np.corrcoef(P.predict(Xte,W)[:,0],y[1600:])[0,1]**2,3))
    return res, round(X.mean()/0.02,1)
if __name__=="__main__":
  for kw in [dict(in_gain=1.2,noise=0.1), dict(in_gain=1.2,noise=0.1,w_e=0.15), dict(in_gain=1.2,noise=0.1,w_e=0.15,g=3.0), dict(in_gain=2.0,noise=0.1,mu=0.7)]:
    print(kw, caps(P.LIF(1,0,**kw)))
