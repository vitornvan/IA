fStepBinComLimiar <- function(net, limiarDaFuncao) {
    if (net >= limiarDaFuncao) {
        return(1)
    } else {
        return(0)
    }
}

fStepBipComLimiar <- function(net, limiarDaFuncao) {
    if (net >= limiarDaFuncao) {
        return(1)
    } else {
        return(-1)
    }
}

perceptron.test <- function(dataset, funcao, weights) {
    classId = ncol(dataset)
    for (i in 1:nrow(dataset)) {
        x = as.numeric(dataset[i, 1:classId-1])
        net = c(x, 1) %*% weights
        yo = funcao(net, 0.5)
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

    sumError = 2 * threshold
    epoca = 0

    while (sumError > threshold) {
        sumError = 0

        for (i in 1:nrow(X)) {
            x = as.numeric(X[i, ])
            y = Y[i]

            net = c(x, 1) %*% weights
            yo = funcao(net, limiarDaFuncao)

            error = y - yo
            sumError = sum(sumError, error^2)

            dE2 = 2 * error * -c(x, 1)

            weights = weights - eta * dE2
        }

        sumError = sumError / nrow(X)

        epoca = sum(epoca, 1)
        cat("Epoca: ", epoca, "\n")
        cat("Pesos: ", weights, "\n")
        cat("Erro quadrático médio: ", sumError, "\n\n")
    }

    cat("Epoca final: ", epoca, "\n")
    cat("Pesos finais: ", weights, "\n")
    cat("Erro quadrático médio final: ", sumError, "\n\n")

    return(weights)
}

execute <- function(dataset, funcao, eta=0.1, limiarDaFuncao = 0.5, threshold = 1e-3) {
    weights = perceptron.train(dataset, funcao, eta, limiarDaFuncao, threshold)
    perceptron.test(dataset, funcao, weights)
}