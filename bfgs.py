
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
        alpha, aval_f = armijo(f, xk, grad_k, dk) # passo alpha calculado a partir das condições de Armijo
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
    # Configuração do retorno
    output = {
        "num_aval_f": num_aval_f,
        "num_aval_grad": num_aval_grad
    }
    
    # Retorno diferenciado em caso de falha de convergência
    if num_iter >= max_iter:
        output = {k: -v for k, v in output.items()}
    
    return output