fStepBinComLimiar <- function(net, limiarDaFuncao) {
    # Se o net for maior ou igual ao limiar, retorna 1, senão retorna 0
    if (net >= limiarDaFuncao) {
        return(1)
    } else {
        return(0)
    }
}

fStepBipComLimiar <- function(net, limiarDaFuncao) {
    # Se o net for maior ou igual ao limiar, retorna 1, senão retorna -1
    if (net >= limiarDaFuncao) {
        return(1)
    } else {
        return(-1)
    }
}

perceptron.test <- function(dataset, funcao, weights) {
    classId = ncol(dataset)

    # Repete para cada linha do dataset de teste
    for (i in 1:nrow(dataset)) {
        # Cria o vetor de entrada referente a linha (x) e o valor esperado (y)
        x = as.numeric(dataset[i, 1:classId-1])

        # Calcula o net com base nos pesos obtidos do treinamento (vêm pelo parâmetro da função)
        net = c(x, 1) %*% weights

        # Usa a função de ativação para calcular o valor obtido do perceptron (yo)
        yo = funcao(net, 0.5)

        # Imprime os resultados
        cat("X:", x, "\nY alcançado:", yo, "\nY esperado:", dataset[i, classId], "\n\n")
    }
}

perceptron.train <- function(dataset, funcao, eta=0.1, limiarDaFuncao = 0.5, threshold = 1e-3) {
    # Implica que a coluna a ser classificada é a última do dataset
    classId = ncol(dataset)

    # X é a matriz de atributos e Y é o vetor de classes
    X = dataset[,1:(classId-1)]	
    Y = dataset[,classId]

    # Inicializa o vetor de pesos (incluindo theta) com valores aleatórios 
    weights = runif(min=-0.5, max = 0.5, n=ncol(X)+1)

    # Inicializa o somatório dos erros e a época
    sumError = 2 * threshold
    epoca = 0

    # Enquanto o erro médio for maior que o limiar desejado
    while (sumError > threshold) {
        sumError = 0

        # Repete para cada linha do dataset de treinamento
        for (i in 1:nrow(X)) {
            # Cria o vetor de entrada referente a linha (x) e o valor esperado (y)
            x = as.numeric(X[i, ])
            y = Y[i]

            # Calcula o net e usa a função de ativação para calcular o valor obtido do perceptron (yo)
            net = c(x, 1) %*% weights
            yo = funcao(net, limiarDaFuncao)

            # Calcula o erro entre o valor obtido (yo) e o esperado (y) e realiza a soma dos quadrados do erro
            error = y - yo
            sumError = sum(sumError, error^2)

            # Calcula a derivada do erro em relação ao peso (dE2) para cada uma das entradas e atualiza os pesos
            dE2 = 2 * error * -c(x, 1)
            weights = weights - eta * dE2
        }

        # Calcula o erro quadrático médio
        sumError = sumError / nrow(X)

        # Incrementa a época
        epoca = sum(epoca, 1)

        # Imprime os resultados da época
        cat("Epoca: ", epoca, "\n")
        cat("Pesos finais da epoca: ", weights, "\n")
        cat("Erro quadrático médio: ", sumError, "\n\n")
    }

    # Imprime os resultados finais
    cat("Epoca final: ", epoca, "\n")
    cat("Pesos finais: ", weights, "\n")
    cat("Erro quadrático médio final: ", sumError, "\n\n")

    return(weights)
}

execute <- function(dataset, funcao, eta=0.1, limiarDaFuncao = 0.5, threshold = 1e-3) {
    # Executa o treinamento e o teste do perceptron
    weights = perceptron.train(dataset, funcao, eta, limiarDaFuncao, threshold)
    perceptron.test(dataset, funcao, weights)
}