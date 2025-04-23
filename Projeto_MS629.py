import pycutest
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

print2 = "1 - Gradiente\n2- BFGS\n3- Espectral\nDigite o numero do algoritmo de descida desejado:"
print3 = "1 - Armijo\n2- Armijo Modificado\n3- Passo Constante\nDigite o numero do tipo de busca desejado:"

search = ["Armijo", "Constante", "Modificada"]

class parameters:
    def __init__(self,function: str,algorithm: str,search: str = "Armijo"):
        self.function = pycutest.import_problem(function)
        self.xk = self.function.x0                                      
        self.old_xk = None
        self.algorithm = algorithm                                       #Algoritmo de descida
        self.grad = self.function.grad(self.xk)                          #sempre teremos uma funcao gradiente para computar
        self.grad_calls = 1                                              #por isso inicializamos a chamada de gradiente como 1
        self.val_calls = 0
        self.old_grad = None
        self.search = search                                             #algoritmo de busca

        #--------------------------------Classe fila para Armijo Modificada--------------------------
        if self.search == "Modificada":
            self.fila = fila(self.function.obj(self.xk))
        else:
            self.fila = None

    def busca(self):
        return self.search
    
    def variation(self):         #retorna delta de x e delta de y
        return  (self.xk - self.old_xk ,self.grad - self.old_grad)  
        
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
        

class fila:
    def __init__(self, x: float):
        self.vector = [None] * 10
        self.atual = 1
        self.vector[0] = x
    def add(self,elem):
        self.vector[self.atual] = elem
        if (self.atual < 9):
            self.atual +=1
        else:
            self.atual = 0 
    def max(self):
        try:
            return max(self.vector)
        except:
            if self.atual == 0:
                return False
            else:
                max = self.vector[0]
                for i in range(self.atual):
                    if self.vector[i] > max:
                        max = self.vector[i]
                return max

def sigma(function: parameters)-> float:
    delta_x, delta_y = function.variation()
    """
    a partir de agora e feito o quadrados minimos
    de uma unica variavel
    com a aproximação da matriz quasi newton
    """
    skyk = np.dot(delta_x,delta_y)             
    if skyk > 0.0:                                                                   #esse if ta no relatorio tem como explicar aqui nao         
        sigma = np.dot(delta_x,delta_y) / np.linalg.norm(delta_x, ord=np.inf)        #caso padrao
    else: 
        sigma = 1.0e-4 * np.linalg.norm(function.grad, ord=np.inf) / max(1.0, np.linalg.norm(function.xk, ord=np.inf))
    return max(1, min(sigma, 1.0e30))

#-------------------------------------------------------------------------------------------------
def direction(parameters: parameters) -> np.array:
    match parameters.algorithm:
        case "Gradiente":
            return -parameters.grad
        case"Espectral":
            return -parameters.grad/sigma(parameters)
        case "BFGS":
            return np.array[1.0,0.0]
#--------------------------------------------------------------------------------------------------
#Condição de Armijo
def Armijo(function: parameters, passo: float, dk, a, b) ->bool:
    aux = function.objective(function.xk + passo*dk)                                          
    if (aux <= passo*a + b):
        if function.busca == "Modificada":
            function.salva_fila(aux)
        return True
    return False
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
    aux = 1e-4*np.dot(parameters.gradient(),direction)                      #parametro eta = 1e-4 referencia relatorio
    for j in range(100):                                                    #limite j = 100 dificilmente chega nele, um numero exagerado de grande
        if Armijo(parameters, passo,direction,aux,M):                       #explicação relatorio
            return passo
        else:
            passo = passo*0.5                                               # parametro beta que está no relatorio a referencia
    return passo

def minimize(function: parameters, tolerance = 1e-5) -> int:
    if function.algorithm == "Gradiente":
        maximum_iterations = int(1e5)                                       #metodo gradiente precisa de mais interaçoes por ter convergencia menor
    else:                                                                   #outros metodos sao mais rapidos ent nao e preciso de tantas interaçoes
        maximum_iterations = int(1e4)
    for iteration in range(maximum_iterations):
        if np.linalg.norm(function.grad) < tolerance:
            return iteration + 1
        else:
            dk = direction(function)
            step = step_parameter(function,dk)
            function.get_new_point(function.xk + step*dk)
    return maximum_iterations

#-----------------------------------------------main-------------------------------------------------------------------------------------------------------
def main():
    problema = testes[int(input())-1]                                       #sera com base na entrada qual funcao minimizar

    descida = algo[int(input(print2))-1]                                    #e qual algoritmo de descida escolher tambem
    if descida == "Espectral":
        busca = search[int(input(print3))-1]                                #e se for Espectral faremos a escolha do algoritmo de busca tambem
        function = parameters(problema,descida,busca)
        aux = -function.gradient()
        function.get_new_point(function.xk + aux*step_parameter(function,aux))
    else:
        function = parameters(problema,descida)
    print(minimize(function))

if __name__ == "__main__":
    main()