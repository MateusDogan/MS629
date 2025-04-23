# Example: using problem classification system
import pycutest
import numpy as np
import time
#começar a usar logging, import logging

testes = [
"ARGLINA", "ARGLINB", "BA-L1SPLS", "BIGGS6", "BROWNAL", "COATING",
"FLETCHCR", "GAUSS2LS", "GENROSE", "HAHN1LS", "HEART6LS", "HILBERTB",
"HYDCAR6LS", "LANCZOS1LS", "LANCZOS2LS", "LRIJCNN1", "LUKSAN12LS",
"LUKSAN16LS", "OSBORNEA", "PALMER1C", "PALMER3C", "PENALTY2", "PENALTY3",
"QING", "ROSENBR", "STRTCHDV", "TESTQUAD", "THURBERLS", "TRIGON1",
"TOINTGOR"]

# 4. Ajuste Dinâmico
#Adaptação Durante a Execução: Comece com valores padrão e ajuste os parâmetros durante a execução, dependendo do progresso. Por exemplo, se você notar que muitos passos estão sendo rejeitados, pode diminuir ββ.

#class otimização:
#    def __init__(self,name):
#        self.problema = pycutest.import_problem(name)
#        self.ponto = self.problema.x0
#    
#    def gradient_norm_sqared(self):
#        return sum([i*i for i in self.problema.grad(self.ponto)])#
#
#    def gradient(self):
#        return self.problema.grad(self.ponto)
#    
#    def atualiza_ponto(self,x):
#        self.ponto = x


p = pycutest.import_problem("ROSENBR")

def armijo(problema,ponto,direcao):
    passo = 1
    k = 1
    b = problema.obj(ponto)
    a = -1e-4*np.dot(direcao,direcao) 
    for k in range(100):
        if problema.obj(ponto - passo*direcao) <= a*passo + b:
            return passo
        else:
            passo *= 0.5
    return passo

def modelo_gradiente(problema, ponto_inicial, erro = 1e-5,limite_iter= 100000):
    iter = 1
    start = time.process_time()
    dk = problema.grad(ponto_inicial)
    passo = armijo(problema,ponto_inicial,dk)
    xk = ponto_inicial - passo*dk
    while iter < limite_iter and np.linalg.norm(problema.grad(xk)) > erro: 
        dk = problema.grad(xk)
        passo = armijo(problema,xk,dk)
        xk = xk - passo*dk
        iter = iter + 1
    end = time.process_time()
    return (problema.obj(xk),iter,(end-start),np.linalg.norm(problema.grad(xk)))

print(modelo_gradiente(p,p.x0))