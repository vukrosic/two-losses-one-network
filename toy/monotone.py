"""Rule checks in every gap x size grid: does P rise with r at each size, and fall with N at each r?
A violation counts only if P drops (or rises) by more than NOISE = 0.15 in P."""
import json, numpy as np
from gapsum import load
NOISE=0.15
SC=["base:1","fine:1","long18k:1","arm:0","armfast:0","weave:1","twoped:1","twoped:2","l1:1","lr3e-3:1","lr3e-4:1",
    "batch64:1","clip250:1","ped12:1","push:1","depth1:1","depth3:1","depth4:1","gru:1","agentnorm:1"]
tot_r=tot_n=bad_r=bad_n=0
for t in SC:
    tag,slow=t.split(":"); own,sh=load(tag,int(slow))
    sizes=[s for s in ["xs","s","sm","m","ml"] if s in own]; A=sorted({a for _,a in sh})
    vr=[];vn=[];nr=nn=0
    for s in sizes:
        P=[sh[(s,a)]["e"]/own[s] for a in A if (s,a) in sh]
        for i in range(len(P)-1):
            for j in range(i+1,len(P)):
                nr+=1
                if P[j]<P[i]-NOISE*max(1,P[i]/2): vr.append("%s r%d<r%d"%(s,j,i))
    learned=[s for s in sizes if own[s]<0.5]
    for a in A:
        P=[sh[(s,a)]["e"]/own[s] for s in learned if (s,a) in sh]
        for i in range(len(P)-1):
            for j in range(i+1,len(P)):
                nn+=1
                if P[j]>P[i]+NOISE*max(1,P[i]/2): vn.append("a%s %s>%s"%(a,learned[j],learned[i]))
    tot_r+=nr;tot_n+=nn;bad_r+=len(vr);bad_n+=len(vn)
    print("%-12s rise-with-gap: %2d/%2d pairs break  %s | fall-with-size (learned sizes %s): %2d/%2d break %s"%(t,len(vr),nr," ".join(vr[:3]),",".join(learned),len(vn),nn," ".join(vn[:3])))
print("TOTAL rise with gap: %d of %d pairs break; fall with size: %d of %d break"%(bad_r,tot_r,bad_n,tot_n))
