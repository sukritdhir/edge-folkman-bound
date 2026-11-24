import sys, time; sys.path.insert(0,'exp')
import numpy as np, qfam, gen
p, e0 = int(sys.argv[1]), int(sys.argv[2])
Q=qfam.QGraph(p,e0); n=Q.n; A=Q.A; np1=p+1; pt=np.arange(n)%np1
def rat(rem_pts, extra=()):
    keep=~np.isin(pt, rem_pts); keep[list(extra)]=False
    idx=np.nonzero(keep)[0]; t=time.time(); r=gen.unweighted_ratio(A[np.ix_(idx,idx)], tol=1e-5); return len(idx), round(r[0],5), r[2], r[3], round(time.time()-t,1)
print('1 vertex', rat([], [0])); sys.stdout.flush()
print('1 fibre', rat([0])); sys.stdout.flush()
print('2 fibre mates (shift 1, 3)', rat([], [0,np1]), rat([],[0,3*np1])); sys.stdout.flush()
print('3 fibre mates', rat([], [0,2*np1,4*np1]), rat([],[0,np1,2*np1])); sys.stdout.flush()
print('2 fibres', rat([0,1]), rat([0,2]), rat([0,np1//2])); sys.stdout.flush()
print('3 fibres', rat([0,1,2]), rat([0,1,np1//2+1])); sys.stdout.flush()
print('4 fibres', rat([0,1,2,3])); sys.stdout.flush()
