# ProvML

Per eseguire il codice:
1. Avviare il server mlflow in un terminale dedicato: `mlflow server`
2. Eseguire lo script di training: `./src/train.py -N 3`

+ Il codice genera un json contenente il grafo generato (`prov_graph.json`)
+ Viene generato anche un file dot, per ottenere l'immagine del grafo: `dot -Tsvg -O prov_graph.dot`