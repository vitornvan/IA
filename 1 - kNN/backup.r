knnTest <- function(datasetTrain, class, query, k=1) {
    # Cria um array para as distâncias
    E = c()

    for (row in 1:ncol(datasetTrain)) {
        soma = 0
        i = 1

        for (col in 1:ncol(datasetTrain)) {
            if (col != class && is.numeric(datasetTrain[row, col]) == TRUE) {
                soma = sum(soma, ((query[i] - datasetTrain[row, col])^2))
                #print("query")
                #print(query[i])
                #print("data value")
                #print(datasetTrain[row, col])
                #print(soma)
                i = i + 1
            }
        }
        E[row] <- sqrt(soma);
    }

    #print(E) 

    ids = sort.list(E)[1:k]
    classes = datasetTrain[ids, class]

    # cat(paste("id: ", ids, "class: ", classes, "dist", E[ids], "\n"))

    U = unique(classes)
    R = rep(0, length(U))

    for (i in 1:length(U)) {
       R[i] = sum(U[i] == classes)
    }

    ret = list()
    ret$U = U
    ret$R = R

    return(classes)
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
        minimo <- min(dataset[, col])
        maximo <- max(dataset[, col])
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
            print(col)
            print(categories)

            for(category in categories) {
                name = paste(col, category, sep = "_")
                dataset[,name] = ifelse(dataset[, col] == category, 1, 0)
            }
        }
    }

    return(dataset)
}

getTrain <- function(dataset) {
    # divide o dataset em parte de 70% para treino e 30% para teste

    datasetTreino <- dataset[1:round(nrow(dataset)*0.7),];

    return(datasetTreino)
}

getTest <- function(dataset) {
    # divide o dataset em parte de 70% para treino e 30% para teste

    datasetTeste <- dataset[(round(nrow(dataset)*0.7)+1):nrow(dataset),];

    return(datasetTeste)
}

execute <- function(dataset, class, k=1) {
    dataset <- removeMissing(dataset)
    dataset <- adjustNumericColumns(dataset)
    dataset <- normalize(dataset)
    dataset <- oneHotEncoding(dataset, class)

    set.seed(713)
  
    n <- nrow(dataset)
    train_indices <- sample(1:n, round(0.7 * n), replace = FALSE)
    train_data <- dataset[train_indices, ]
    test_data <- dataset[-train_indices, ]

    #print(nrow(train_data))
    #print(nrow(test_data))

    answer = NULL

    soma = 0

    for (row in 1:nrow(test_data)) {
        
        arr = as.numeric(test_data[row, ])
        arr = na.omit(arr)
        answer <- test_data[row, class]

        cat(paste("Teste", row, ":\n"))
        cat(paste("Resposta esperada: ", answer, "\n"))
        resp <- knnTest(train_data, class, arr, k)
        cat(paste("Resposta obtida: ", resp, "\n"))


        freq <- table(resp)
        most_freq_index <- which.max(freq)
        most_freq_value <- names(freq)[most_freq_index]
        print(most_freq_value)


        if (answer == most_freq_value) soma = sum(soma, 1)
    }
    print(paste("Acertos: ", soma, " de ", nrow(test_data), " ( acurácia de", soma/nrow(test_data)*100, "%)"))
}