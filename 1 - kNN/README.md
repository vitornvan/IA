#KNN

## Para executar, abra o R no terminal na pasta "1 - kNN" e use:

```R
source("source.r")
dataset <- read.csv("crx.data", header = FALSE)
execute(dataset, colunaQueSeQuerPrever, kVizinhosProximosDesejados)
```

## Para executar pra todas as colunas categóricas, no mesmo local, use:

```R
source("source.r")
dataset <- read.csv("crx.data", header = FALSE)
executeAllTests(dataset, kVizinhosProximosDesejados, c(1, 4, 5, 6, 7, 9, 10, 12, 13, 16))
```