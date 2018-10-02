import pandas as pd
import statistics
data = pd.read_csv("saida-artigo.csv",delimiter=";")



teste = list(data.c.values)

teste = [float(elemento) for elemento in teste]


print("Moda -------------------------")
print(statistics.mode(teste))

print("Mediana -----------------------------")
print(statistics.median(teste))
print("Variancia ------------------------------")

print(statistics.variance(teste))
print("Desvio padrão ---------------------------")
print(statistics.stdev(teste))

print("Média ---------------------------")
print(statistics.mean(teste))

print(data)
