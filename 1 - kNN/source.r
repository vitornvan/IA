knnTest <- function(datasetTrain, class, query, k=1) {
    # Cria um array para as distâncias euclidianas entre cada um dos pontos do dataset de treino e a query a ser classificada
    E = c()

    # Para cada linha do dataset de treino, calcula a distância euclidiana entre o valor da query e o valor do dataset de treino
    for (row in 1:nrow(datasetTrain)) {
        soma = 0
        i = 1

        for (col in 1:ncol(datasetTrain)) {
            if (col != class && is.numeric(datasetTrain[row, col]) == TRUE) { # Apenas para colunas numéricas e que não sejam a coluna da classe
                soma = sum(soma, ((as.numeric(query[i]) - as.numeric(datasetTrain[row, col]))^2))
                i = i + 1
            }
        }
        E[row] <- sqrt(soma);
    }

    ids = sort.list(E)[1:k] # Identifica os k vizinhos mais próximos e seus ids
    classes = datasetTrain[ids, class] # Identifica as classes dos k vizinhos mais próximos, com base em seus ids e na coluna da classe

    U = unique(classes) # Identifica as classes únicas
    R = rep(0, length(U)) 

    # Identifica as repetições das classes únicas
    for (i in 1:length(U)) {
       #R[i] = sum(U[i] == classes)
    }

    ret = list()
    ret$U = U
    ret$R = R

    return(classes) # Retorna as classes dos k vizinhos mais próximos
}

removeMissing <- function(dataset) { 
  # Transforma todos os valores ? em NA reconhecido pelo R
  for(col in 1:ncol(dataset)) {
    dataset[, col] <- replace(dataset[, col], dataset[, col] == "?", NA)
  }
  
  # Deleta todos os valores NA
  dataset <- na.omit(dataset)
  return(dataset)
}

normalize <- function(dataset) {
    # Normaliza os valores das colunas pela técnica de reescala linear
    for(col in 1:ncol(dataset)) {
        # identifica o mínimo e o máximo da coluna
        minimo <- min(as.numeric(dataset[, col]))
        maximo <- max(as.numeric(dataset[, col]))
        for (row in 1:nrow(dataset)) {
            if (is.numeric(dataset[row, col]) == TRUE)
            # para cada valor numérico, aplica a fórmula de normalização
            dataset[row, col] <- (dataset[row, col] - minimo) / (maximo - minimo)
        }
    }
    return(dataset)
}

adjustNumericColumns <- function(dataset) {
    # Transforma colunas com as classes definidas de forma incorreta
    # segundo a descrição dos dados, V2 e V14 são numéricas, porém estavam sendo indicadas como categóricas
    dataset$V2 <- as.numeric(dataset$V2)
    dataset$V14 <- as.numeric(dataset$V14)

    return(dataset)
}

oneHotEncoding <- function(dataset, knnClassColumn) {
    # Divide as colunas categóricas em colunas numéricas usando one hot encoding
    # Não divide a coluna da classe a ser prevista pelo KNN (da mesma forma usada no vídeo exemplo) 
    numColunasInicial = ncol(dataset)

    for(col in 1:numColunasInicial) {
        if (is.character(dataset[, col]) && col != knnClassColumn) {
            categories = unique(dataset[, col])

            for(category in categories) {
                name = paste(col, category, sep = "_")
                dataset[,name] = ifelse(dataset[, col] == category, 1, 0)
            }
        }
    }

    return(dataset)
}

execute <- function(dataset, class, k=1) {
    dataset <- removeMissing(dataset) # Remove as linhas de valores faltantes (?)
    dataset <- adjustNumericColumns(dataset) # Corrige classificação das colunas que deveriam ser numéricas
    dataset <- normalize(dataset) # Normaliza os dados numéricos
    dataset <- oneHotEncoding(dataset, class) # Divide as colunas categóricas em colunas numéricas usando one hot encoding

    set.seed(713) # Seed aleatória para permitir reprodutibilidade da separação dos dados de treino e teste
  
    # Divide os dataset em dados de treino e dados de teste
    n <- nrow(dataset)
    indicesDeTreino <- sample(1:n, round(0.7 * n), replace = FALSE)
    train_data <- dataset[indicesDeTreino, ]
    test_data <- dataset[-indicesDeTreino, ]

    answer = NULL
    somaCorretos = 0

    # Executa cada linha do dataset de teste com a função KNN
    for (row in 1:nrow(test_data)) {
        
        # Separa as linhas em um array para serem passadas para a função KNN (apenas com os dados numéricos)
        arr = as.numeric(test_data[row, ])
        arr = na.omit(arr)
        # Identifica a resposta correta para a linha do dataset de teste (label)
        answer <- test_data[row, class]

        cat(paste("Teste", row, ":\n"))
        cat(paste("Resposta esperada: ", answer, "\n"))

        # Realiza o teste com a função KNN
        resp <- knnTest(train_data, class, arr, k)
        cat("Respostas obtidas: ")
        cat(resp, "\n")

        # Identifica a resposta mais frequente, ou seja, a resposta que mais aparece entre os k vizinhos mais próximos
        freq <- table(resp)
        indexMaisFrequente <- which.max(freq)
        valorMaisFrequente <- names(freq)[indexMaisFrequente]
        cat(paste("Classe mais frequente: ", valorMaisFrequente, "\n\n"))

        # Se a resposta vor correta, adiciona 1 ao contador de acertos
        if (answer == valorMaisFrequente) somaCorretos = sum(as.numeric(somaCorretos), as.numeric(1))
    }

    # Imprime a quantidade de acertos x quantidade total de testes e calcula a acurácia
    print(paste("Acertos: ", somaCorretos, " de ", nrow(test_data), " (acurácia de", somaCorretos/nrow(test_data)*100, "%)"))

    return (somaCorretos/nrow(test_data)*100);
}

executeAllTests <- function(dataset, k, columnsForTests) {
    somaAcuracias = 0;
    for (i in columnsForTests) {
        acuracia = execute(dataset, i, k)

        somaAcuracias = sum(as.numeric(somaAcuracias), as.numeric(acuracia))
    }

    print(paste("Acurácia média: ", somaAcuracias/length(columnsForTests), "%"))

}
