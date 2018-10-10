import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("saida-artigo.csv", delimiter=";")

print(data.sort_values("c")) 
