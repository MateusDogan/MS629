#import pycutest

import numpy as np
import time

testes = [
"ARGLINA", "ARGLINB", "BA-L1SPLS", "BIGGS6", "BROWNAL", "COATING",
"FLETCHCR", "GAUSS2LS", "GENROSE", "HAHN1LS", "HEART6LS", "HILBERTB",
"HYDCAR6LS", "LANCZOS1LS", "LANCZOS2LS", "LRIJCNN1", "LUKSAN12LS",
"LUKSAN16LS", "OSBORNEA", "PALMER1C", "PALMER3C", "PENALTY2", "PENALTY3",
"QING", "ROSENBR", "STRTCHDV", "TESTQUAD", "THURBERLS", "TRIGON1",
"TOINTGOR"]

algo = ["Gradiente", "BFGS", "Espectral"]

print1 = "1 - ARGLINA\n2 - ARGLINB\n3 - BA-L1SPLS\n4 - BIGGS6\n5 - BROWNAL\n6 - COATING\n7 - FLETCHCR\n8 - GAUSS2LS\n9 - GENROSE\n10 - HAHN1LS\n11 - HEART6LS\n12 - HILBERTB\n13 - HYDCAR6LS\n14 - LANCZOS1LS\n15 - LANCZOS2LS\n16 - LRIJCNN1\n17 - LUKSAN12LS\n18 - LUKSAN16LS\n19 - OSBORNEA\n20 - PALMER1C\n21 - PALMER3C\n22 - PENALTY2\n23 - PENALTY3\n24 - QING\n25 - ROSENBR\n26 - STRTCHDV\n27 - TESTQUAD\n28 - THURBERLS\n29 - TRIGON1\n30 - TOINTGOR\nDigite o número do problema de teste desejado:"
print2 = "1 - Gradiente\n2 - BFGS\n3 - Espectral\nDigite o numero do algoritmo de descida desejado:"
print3 = "1 - Armijo\n2 - Armijo Modificado\n3 - Passo Constante\nDigite o numero do tipo de busca desejado:"

search = ["Armijo", "Constante", "Modificada"]

class parameters:
    def __init__(self,function: str,algorithm: str,search: str = "Armijo"):
        self.function = pycutest.import_problem(function)                #deixar so os metodos publicos depois
        self.xk = self.function.x0                                      
        self.old_xk = None
        self.algorithm = algorithm                                       #Algoritmo de descida
        self.grad = self.function.grad(self.xk)                          #sempre teremos uma funcao gradiente para computar
        self.grad_calls = 1                                              #por isso inicializamos a chamada de gradiente como 1
        self.val_calls = 0
        self.old_grad = None
        self.search = search                                             #algoritmo de busca

        #--------------------------------Espectral--------------------------------------------------
        if self.algorithm == "Espectral":
            aux = -self.gradient()
            new_xk = self.xk + step_parameter(self,aux) * aux
            self.get_new_point(new_xk)

        #--------------------------------Classe fila para Armijo Modificada--------------------------
        if self.search == "Modificada":
            self.fila = Fila(self.objective(self.xk))
            self.fila.add()
        else:
            self.fila = None

    def busca(self):
        return self.search
        
    def get_new_point(self,x):
        self.old_xk = self.xk
        self.xk = x
        self.old_grad = self.grad
        self.grad = self.function.grad(self.xk)
        self.grad_calls += 1
        
    def gradient(self):
        return self.grad
    
    def modified(self):
      if self.fila != None:
          return self.fila.max() 
    
    def objective(self, ponto):
        self.val_calls += 1
        return self.function.obj(ponto)
    
    def salva_fila(self, valor)-> None:
        self.fila.add(valor)
    
    def variation(self):         #retorna delta de x e delta de y
        return  (self.xk - self.old_xk ,self.grad - self.old_grad)  
        

class Fila:
    def __init__(self, x: float):
        self.vector = [None] * 10                  #classe fila foi criada a fim de ser usada no gradiente espectral
        self.atual = 1                             #pode ser usada como heuristica nos outros metodos
        self.vector[0] = x                         #mas nao sera feito isso nesse relatorio
    
    def add(self, elem):
        self.vector[self.atual] = elem
        self.atual = (self.atual + 1) % 10         #implementa rotação

    def first(self):
        return self.vector[0]
    
    def max(self):
        elementos_validos = [v for v in self.vector if v is not None]
        return max(elementos_validos) if elementos_validos else None 

#------------------------------------------------------

def bfgs(f, xk, tol = 1.0e-5, max_iter = 10000):
    '''Dado um ponto inicial xk de uma função f, este método tenta encontrar um mínimo local aplicando o algoritmo BFGS 
    por no máximo 10.000 iterações, ou até que a norma-infinito do gradiente seja menor ou igual ao valor de tolerância 1.0e-5'''
    Hk = I = np.eye(len(xk)) # matriz identidade utilizada como a aproximação inicial para a inversa da Hessiana
    num_iter = 0
    num_aval_f = 0
    num_aval_grad = 0
    start = time.process_time()
    grad_k = f.grad(xk) # gradiente no ponto xk
    num_aval_grad += 1
    norm_k = np.linalg.norm(grad_k, ord = np.inf) # norma-infinito do gradiente
    while (num_iter < max_iter) and (norm_k > tol): 
        dk = Hk @ (-1*grad_k) # direção de descida dada pelo produto entre a matriz Hk e o gradiente no ponto xk
        alpha, aval_f = armijo_bfgs(f, xk, grad_k, dk) # passo alpha calculado a partir das condições de Armijo
        num_aval_f += aval_f
        xk_new = xk + alpha*dk # valor do ponto é atualizado no sentido oposto ao gradiente
        # cálculo do delta x e da variação do gradiente, utilizados para atualização da matriz Hk:
        sk = xk_new - xk
        yk = f.grad(xk_new) - grad_k
        num_aval_grad += 1
        # se o produto interno skyk é positivo, atualiza-se a matriz Hk através da fórmula a seguir, caso contrário mantém-se a mesma aproximação para a próxima iteração
        if np.dot(sk, yk) > 0.0:
            Vk = I - np.outer(sk, yk)/np.dot(sk, yk)
            Hk = Vk @ Hk @ Vk.T + np.outer(sk, sk)/np.dot(sk, yk)
        xk = xk_new # atualização do ponto
        grad_k = f.grad(xk) # atualização do gradiente
        num_aval_grad += 1
        norm_k = np.linalg.norm(grad_k, ord = np.inf) # atualização da norma
        num_iter += 1
    valor = f.obj(xk) # valor da função no último ponto avaliado (possível mínimo local)
    num_aval_f += 1
    end = time.process_time()
    tempo = end - start # tempo de execução do algoritmo
    xk_formatado = np.array2string(xk, formatter={'all': lambda x: f"{x:.2e}"}) # formatação dos valores do vetor xk
    if (num_iter < max_iter):
        log = "Algoritmo concluído com sucesso."
    else:
        log = "Número máximo de iterações atingido."
    output = (f"{log}\n"
        f"Valor: {valor:.2e}, "
        f"Ponto: {xk_formatado:}, "
        f"Norma: {norm_k:.2e}, "
        f"Iterações: {num_iter:}, "
        f"Avaliações da função: {num_aval_f:}, "
        f"Avaliações do gradiente: {num_aval_grad:}, "
        f"Tempo: {tempo:.4f}s")
    return output

def sigma(function: parameters)-> float:
    deltax, deltay =  function.variation()         # delta de y
    """
    a partir de agora e feito o quadrados minimos
    de uma unica variavel
    com a aproximação da matriz quasi newton
    """
    skyk = np.dot(deltax,deltay)
    if skyk > 0.0:                                      
        sigma = np.dot(deltax,deltay) / np.dot(deltax,deltax)
    else: 
        sigma = 1.0e-4 * np.linalg.norm(function.grad, ord=np.inf) / max(1.0, np.linalg.norm(function.xk, ord=np.inf))
    return max(1.0e-30, min(sigma, 1.0e30))

#-------------------------------------------------------------------------------------------------
def direction(parameters: parameters) -> np.array:
    match parameters.algorithm:
        case "Gradiente":
            return -parameters.gradient()
        case"Espectral":
            return -parameters.gradient()/sigma(parameters)
        
#--------------------------------------------------------------------------------------------------
#Condição de Armijo
def Armijo(function: parameters, passo: float, dk, alpha, beta) ->bool:
    aux = function.objective(function.xk + passo*dk)                            #utilizar armijo so para verificar                        
    if (aux <= passo*alpha + beta):                                             #se o passo e bom ou nao
        if function.busca == "Modificada":
            function.salva_fila(aux)
        return True
    return False
#--------------------------------------------------------------------------------------------------
#Condição de Armijo para o caso BFGS
def armijo_bfgs(f, xk, grad_k, dk, v = 0, alpha = 1, beta = 0.5, eta = 1.0e-4, max_iter = 100):
    num_iter = 0
    aval_f = 0
    a = eta*np.dot(grad_k, dk)
    c = f.obj(xk + alpha*dk)
    aval_f += 1
    if (v == 0): # caso padrão
        b = f.obj(xk)
        aval_f += 1
    else: # caso do armijo modificado para o método espectral
        b = v.max() 
    while (num_iter < max_iter) and (c > b + a*alpha): # fórmula da condição de armijo
        alpha = alpha*beta # atualização do passo
        c = f.obj(xk + alpha*dk) # atualização do parâmetro c
        aval_f += 1
        num_iter += 1
    if (v != 0):
        v.add(c)
    return alpha, aval_f
#--------------------------------------------------------------------------------------------------
#Busca usada
def step_parameter(parameters: parameters, direction) -> float:
    if parameters.busca() == "Constante":
        return 1.0
    #------------------------Passos nao constantes----------------
    if parameters.busca() == "Modificada":
        M = parameters.modified()
    else:
        M = parameters.objective(parameters.xk)
    passo = 1.0
    aux = 1e-4*np.dot(parameters.grad,direction)                            #parametro eta = 1e-4 referencia relatorio
    for j in range(100):                                                    #limite j = 100 dificilmente chega nele, um numero exagerado de grande
        if Armijo(parameters, passo,direction,aux,M):                       #explicação detalhada esta no relatorio
            return passo
        else:
            passo = passo*0.5                                               #parametro beta que está no relatorio a referencia
    return passo                                                            #mas qualquer numero entre 0,5 a 0,8 converge bem

def minimize(function: parameters, tolerance = 1e-4):
    if function.algorithm == "BFGS":
        return bfgs(function.function, function.xk)                         #metodos do tipo gradiente precisam de mais interaçoes                                                               
    maximum_iterations = int(1e5)                                           #BFGS pode usar menos iteraçoes
    start = time.perf_counter()
    for iteration in range(maximum_iterations):
        if np.linalg.norm(function.grad, ord=np.inf) < tolerance:
            tempo = start - time.perf_counter()
            return {
                "val_calls": function.val_calls,
                "grad_calls": function.grad_calls,
                "tempo": tempo
            }
        else:
            dk = direction(function)
            step = step_parameter(function,dk)
            function.get_new_point(function.xk + step*dk)
    return maximum_iterations

#-----------------------------------------------main-------------------------------------------------------------------------------------------------------
def main():
    problema = testes[int(input(print1))-1]                                 #sera com base na entrada qual funcao minimizar

    descida = algo[int(input(print2))-1]                                    #e qual algoritmo de descida escolher tambem
    if descida == "Espectral":                                              #No cado do metodo gradiente Espectral 
        busca = search[int(input(print3))-1]                                #Sera feitoa a escolha do algoritmo de busca
        function = parameters(problema,descida,busca)                       #Sera feito direto a escolha de x1 antes de iniciar minimize
    else:
        function = parameters(problema,descida)
    print(minimize(function))

if __name__ == "__main__":
    main()