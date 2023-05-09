fSigmoid <- function(net) {
  return(1/(1+exp(-net)))
}

dfSigmoid <- function(fNet) {
  return(fNet*(1-fNet))
}

fTanH <-function(net) {
    return((2/(1+exp(-2*net)))-1)
}

dfTanH <- function(fNet) {
    return((1-fNet)^2) # Verificar essa derivada
}

normalize <- function(dataset) {
    # Normaliza os valores das colunas pela técnica de reescala linear
    for(col in 1:ncol(dataset)) {
        # identifica o mínimo e o máximo da coluna
        minimo <- min(as.numeric(dataset[, col]))
        maximo <- max(as.numeric(dataset[, col]))
        for (row in 1:nrow(dataset)) {
            if (!is.character(dataset[row, col]) == TRUE)
            # para cada valor numérico, aplica a fórmula de normalização
            dataset[row, col] <- (as.numeric(dataset[row, col]) - minimo) / (maximo - minimo)
        }
    }
    return(dataset)
}

oneHotEncoding <- function(dataset, classColumn) {
    # Divide as colunas categóricas em colunas numéricas usando one hot encoding
    # Não divide a coluna da classe a ser classificada
    numColunasInicial = ncol(dataset)

    for(col in 1:numColunasInicial) {
        if (!is.numeric(dataset[, col]) && col != classColumn) {
            categories = unique(dataset[, col])

            for(category in categories) {
                name = paste(col, category, sep = "_")
                dataset[,name] = ifelse(dataset[, col] == category, 1, 0)
            }
        }
    }

    return(dataset)
}

mlp.architecture <- function(
    input.lenght=2,
    hidden.lenght=2,
    output.lenght=1,
    activation.function=fSigmoid,
    dactivation.function=dfSigmoid) {

    model = list()
    model$input.lenght = input.lenght
    model$hidden.lenght = hidden.lenght
    model$output.lenght = output.lenght
    
    model$hidden = matrix(runif(
        min=-0.5,
        max=0.5,
        n=(input.lenght+1)*hidden.lenght
    ), nrow=hidden.lenght, ncol=input.lenght+1)

    model$output = matrix(runif(
        min=-0.5,
        max=0.5,
        n=(hidden.lenght+1)*output.lenght
    ), nrow=output.lenght, ncol=hidden.lenght+1)

    model$f = activation.function
    model$dfDNet = dactivation.function

    return(model)
}

mlp.forward <- function(model, Xp) {

    #hidden
    netHP = model$hidden %*% c(Xp, 1)
    fNetHP = model$f(netHP)

    #output
    netOP = model$output %*% c(as.numeric(fNetHP), 1)
    fNetOP = model$f(netOP)

    ret = list()
    ret$netHP = netHP
    ret$fNetHP = fNetHP
    ret$netOP = netOP
    ret$fNetOP = fNetOP

    return(ret)
}

mlp.backpropagation <- function(model, datasetTrain, eta=0.1, treshold=1e-3) {
    
    squaredError = 2 * treshold
    counter = 1

    while(squaredError > treshold) {
        squaredError = 0

        for (p in 1:nrow(datasetTrain)) {
            Xp = as.numeric(datasetTrain[p, 1:model$input.lenght])
            Yp = as.numeric(datasetTrain[p, (as.numeric(model$input.lenght)+1):ncol(datasetTrain)])
            
            results = mlp.forward(model, Xp)
            Op = results$fNetOP

            # Calculando erro
            error = Yp - Op

            squaredError = sum(squaredError, sum(error^2))

            deltaOP = error * model$dfDNet(results$fNetOP)

            wOkj = model$output[, 1:model$hidden.lenght]

            deltaHP = as.numeric(model$dfDNet(results$fNetHP)) * (as.numeric(deltaOP) %*% wOkj)
            #print(Yp)
            #print(Op)
            #print(as.vector(c(results$fNetHP, 1)))

            model$output = model$output + eta * (deltaOP %*% as.vector(c(results$fNetHP, 1)))

            model$hidden = model$hidden + eta * (t(deltaHP) %*% as.vector(c(Xp, 1)))

        }

        squaredError = squaredError / nrow(datasetTrain)
        cat("Epoca: ", counter, " - Erro medio quadrado: ", squaredError, "\n")
        counter = sum(counter, 1)
    }

    ret = list()

    ret$model = model
    ret$counter = counter

    return(ret)
}

execute <- function(dataset, eta=0.1, treshold=1e-3) {
    dataset <- dataset[-c(1001:10000),]

    dataset = oneHotEncoding(dataset, 1)
    dataset = normalize(dataset)

    dataset = dataset[, -c(13, 14, 15)]
    print(dataset[1, 1:ncol(dataset)])


    model = mlp.architecture(12, 25, 1)

     # Divide os dataset em dados de treino e dados de teste
    n <- nrow(dataset)
    indicesDeTreino <- sample(1:n, round(0.7 * n), replace = FALSE)
    train_data <- dataset[indicesDeTreino, ]
    test_data <- dataset[-indicesDeTreino, ]


    trained = mlp.backpropagation(model, train_data, eta, treshold)

    trainedModel = trained$model
    epocaFinal = trained$counter

    acertos = 0;

    # Testando o modelo
    for (p in 1:nrow(test_data)) {
        Xp = as.numeric(test_data[p, 1:trainedModel$input.lenght])
        Yp = as.numeric(test_data[p, (as.numeric(trainedModel$input.lenght)+1):ncol(test_data)])

        results = mlp.forward(trainedModel, Xp)
        Op = results$fNetOP

        if (Op > 0.5) {
            Op = 1
        } else {
            Op = 0
        }

        cat("Teste: ", p, "\n")
        cat("Valor esperado: ", Yp, " - Valor Obtido: ", Op, "\n\n")

        if (Op == Yp) acertos = sum(acertos, 1)
    }

    print(paste("Resultados da época final: ", epocaFinal))
    print(paste("Acertos: ", acertos, " de ", nrow(test_data), " (acurácia de", acertos/nrow(test_data)*100, "%)"))
}

executeTanH <- function(dataset, eta=0.1, treshold=1e-3) {
    dataset <- dataset[-c(1001:10000),]

    dataset = oneHotEncoding(dataset, 1)
    dataset = normalize(dataset)

    dataset = dataset[, -c(13, 14, 15)]
    dataset[, ncol(dataset)] <- replace(dataset[, ncol(dataset)], dataset[, ncol(dataset)] == "0", -1)

  

    print(dataset[1, 1:ncol(dataset)])
    print(dataset[2, 1:ncol(dataset)])
    print(dataset[3, 1:ncol(dataset)])


    model = mlp.architecture(12, 25, 1, fTanH, dfTanH)

     # Divide os dataset em dados de treino e dados de teste
    n <- nrow(dataset)
    indicesDeTreino <- sample(1:n, round(0.7 * n), replace = FALSE)
    train_data <- dataset[indicesDeTreino, ]
    test_data <- dataset[-indicesDeTreino, ]


    trained = mlp.backpropagation(model, train_data, eta, treshold)

    trainedModel = trained$model
    epocaFinal = trained$counter

    acertos = 0;

    # Testando o modelo
    for (p in 1:nrow(test_data)) {
        Xp = as.numeric(test_data[p, 1:trainedModel$input.lenght])
        Yp = as.numeric(test_data[p, (as.numeric(trainedModel$input.lenght)+1):ncol(test_data)])

        results = mlp.forward(trainedModel, Xp)
        Op = results$fNetOP

        if (Op > 0.5) {
            Op = 1
        } else {
            Op = 0
        }

        cat("Teste: ", p, "\n")
        cat("Valor esperado: ", Yp, " - Valor Obtido: ", Op, "\n\n")

        if (Op == Yp) acertos = sum(acertos, 1)
    }

    print(paste("Resultados da época final: ", epocaFinal))
    print(paste("Acertos: ", acertos, " de ", nrow(test_data), " (acurácia de", acertos/nrow(test_data)*100, "%)"))
}
