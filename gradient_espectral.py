import pycutest
import numpy as np
import time

testes = [
"ARGLINA", "ARGLINB", "BA-L1SPLS", "BIGGS6", "BROWNAL", "COATING",
"FLETCHCR", "GAUSS2LS", "GENROSE", "HAHN1LS", "HEART6LS", "HILBERTB",
"HYDCAR6LS", "LANCZOS1LS", "LANCZOS2LS", "LRIJCNN1", "LUKSAN12LS",
"LUKSAN16LS", "OSBORNEA", "PALMER1C", "PALMER3C", "PENALTY2", "PENALTY3",
"QING", "ROSENBR", "STRTCHDV", "TESTQUAD", "THURBERLS", "TRIGON1", "TOINTGOR"]


p = pycutest.import_problem(testes[0])

def armijo(problema,ponto,direcao,grad,beta = 0.5,eta = 1e-4,limite = 100):
    passo = 1
    k = 1
    b = problema.obj(ponto)
    a = eta*np.dot(grad,direcao) 
    for k in range(limite):
        if problema.obj(ponto + passo*direcao) <= a*passo + b:
            return passo
        else:
            passo *= beta
    return passo

def grad_espc(problema,x0,limite = 100000, tol = 1e-5):
    g = problema.grad(x0)
    x1 = x0 - g*armijo(problema,x0,-g,g)

    for iter in range(limite):
        a = np.linalg.norm(problema.grad(x1))
        print(a)
        print(iter)
        if a < tol:
            return a
        else:
            gk = problema.grad(x1)
            varx = x1 - x0
            vary = gk - problema.grad(x0)
            skyk = np.dot(varx,vary)
            if skyk > 0.0:
                sigma = np.dot(varx,vary) / np.linalg.norm(varx, ord=np.inf)
            else:
                sigma = 1.0e-4 * np.linalg.norm(g) / max(1.0, np.linalg.norm(x1))
            sigma = max(1.0e-30, min(sigma, 1.0e30))

            x0 = x1
            x1 = x1 - armijo(problema,x1,-gk/sigma,gk)*gk/sigma
            
print(grad_espc(p,p.x0))

