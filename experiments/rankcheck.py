# Shared rank-k linear model, best possible fit (Eckart-Young): keeps the k largest loss-weighted directions.
# Each thing has a power-law spectrum c * i^-beta. Own models get k/2 directions each.
import numpy as np
M=200000
for beta in [2,3,4]:
  for k in [40,400]:
    print(f"beta {beta} k {k}")
    for lam in [3,10,100,1000]:
      i=np.arange(1,M+1.)
      f=lam*i**-beta; s=i**-beta
      own_f=f[k//2:].sum(); own_s=s[k//2:].sum()
      L=own_f/own_s                      # loss ratio from own models
      allv=np.r_[f,s]; tag=np.r_[np.zeros(M),np.ones(M)]
      keep=np.argsort(-allv)[:k]
      ns=int(tag[keep].sum())
      P=s[ns:].sum()/own_s
      rho=L**(1/beta)
      print(f"  Lambda {L:7.1f}  slow gets {ns:3d}/{k}  P {P:7.2f}  formula ((1+rho)/2)^(beta-1) {((1+rho)/2)**(beta-1):7.2f}")
