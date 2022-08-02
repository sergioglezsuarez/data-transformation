from preprocesado import codificadorEtiqueta, escalar, validacion_modelo
import numpy as np
import pandas as pd

# casos de prueba

codificador = codificadorEtiqueta.CodificadorEtiqueta()
lista = ["agua", "tierra", "agua", "aire", "aire", "fuego"]
codificador.ajustar(lista)
print(codificador.transformar(np.array(lista)))
print(codificador.transformar_inv(pd.DataFrame([0, 1, 0, 1, 2, 3])))

escalador = escalar.Escalar()
lista = [[1, 10], [2, 20]]
a = np.array(lista)
b = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [3, .9, 6, 35, 88]})
escalador.ajustar(0, 100)
print(escalador.transformar(a))
print(escalador.transformar(b))
print(escalador.transformar(lista))
print(escalador.transformar_inv(pd.DataFrame([[0, 0], [20, 20], [40, 40], [60, 60], [80, 80], [100, 100]])))

np.random.seed(42)
X = np.random.randint(0, high=20, size=(50, 4))
y = ['A'] * 45 + ['B'] * 5
df = pd.DataFrame(X, y)
print(df)
print(validacion_modelo.divide_entrenamiento_test(X, tam_train=25, mezclar=True, semilla=56, balancear=y))
print(validacion_modelo.divide_entrenamiento_test(X, tam_train=0.75, mezclar=False, semilla=None, balancear=None))
