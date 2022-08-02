import numpy as np
import pandas as pd


class CodificadorEtiqueta:
    """
    Clase que cuenta con tres métodos para convertir datos simbólicos a numéricos.

    Atributos:
        count (int): contador de los valores numéricos usados.
        dict (dict): diccionario con las equivalencias entre valores simbólicos y numéricos.
        dict_inv (dict): diccionario con las equivalencias entre valores numéricos y simbólicos.
    """
    def __init__(self):
        """
        Función constructora de la clase.
        """

        self.count = 0
        self.dict = {}
        self.dict_inv = {}

    def ajustar(self, y):
        """
        Función que obtiene las equivalencias entre valores simbólicos y numéricos y viceversa.

        :param y: (list/numpy.array/pandas.DataFrame) iterable unidimensional con los valores simbólicos a convertir.
        """

        if y is None:
            raise ValueError("No se ha suministrado una estructura de datos.")

        dataType = type(y)

        # es más fácil trabajar con listas así que convertimos los datos si es necesario
        if dataType == list:
            data = y
        elif dataType.__module__ == "numpy":
            data = y.tolist()
        elif dataType.__module__ == "pandas.core.frame":
            data = y.values.tolist()
        else:
            raise TypeError("Estructura de datos no reconocida.")

        self.dict = dict.fromkeys(data)  # creamos un diccionario con los elementos del iterable como claves
        for i in self.dict:  # le asignamos un número como valor a cada clave y a la vez creamos el diccionario inverso
            self.dict[i] = self.count
            self.dict_inv[self.count] = i
            self.count += 1

    def transformar(self, y):
        """
        Función que convierte valores simbólicos a numéricos conforme las equivalencias obtenidas en ajustar.

        :param y: (list/numpy.array/pandas.DataFrame) iterable unidimensional con valores simbólicos.
        :return: (list/numpy.array/pandas.DataFrame) iterable unidimensional con los valores numéricos correspondientes.
        """

        if y is None:
            raise ValueError("No se ha suministrado una estructura de datos.")

        dataType = type(y)
        if dataType == list:
            data = y
        elif dataType.__module__ == "numpy":
            data = y.tolist()
        elif dataType.__module__ == "pandas.core.frame":
            data = y[0].values.tolist()
        else:
            raise TypeError("Estructura de datos no reconocida.")

        if any([i not in self.dict for i in data]):  # si hay algún valor que no pertenezca a la codificación se muestra error
            raise ValueError("No todos los valores pertenecen a la codificación.")
        else:
            numericData = [self.dict[i] for i in data]  # obtenemos el equivalente de cada valor del diccionario

        if dataType == list:
            return numericData
        elif dataType.__module__ == "numpy":
            return np.array(numericData)
        elif dataType.__module__ == "pandas.core.frame":
            return pd.DataFrame(numericData)

    def transformar_inv(self, y):
        """
        Función que convierte valores numéricos a simbólicos conforme las equivalencias obtenidas en ajustar.

        :param y: (list/numpy.array/pandas.DataFrame) iterable unidimensional con valores numéricos.
        :return: (list/numpy.array/pandas.DataFrame) iterable unidimensional con valores simbólicos.
        """

        if y is None:
            raise ValueError("No se ha suministrado una estructura de datos.")

        dataType = type(y)
        if dataType == list:
            data = y
        elif dataType.__module__ == "numpy":
            data = y.tolist()
        elif dataType.__module__ == "pandas.core.frame":
            data = y[0].values.tolist()
        else:
            raise TypeError("Estructura de datos no reconocida.")

        if any([i not in self.dict_inv for i in data]):
            raise ValueError("No todos los valores pertenecen a la codificación.")
        else:
            symbolicData = [self.dict_inv[i] for i in data]

        if dataType == list:
            return symbolicData
        elif dataType.__module__ == "numpy":
            return np.array(symbolicData)
        elif dataType.__module__ == "pandas.core.frame":
            return pd.DataFrame(symbolicData)
