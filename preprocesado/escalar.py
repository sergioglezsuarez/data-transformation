import numpy as np
import pandas as pd


class Escalar:
    """
    Clase que cuenta con tres métodos para escalar valores numéricos a un rango dado.

    Atributos:
        min (int/float): límite inferior del rango al cual escalar.
        max (int/float): límite superior del rango al cual escalar.
        constants (list): partes constantes de la fórmula de escalado.
    """
    def __init__(self):
        """
        Función constructora de la clase.
        """

        self.min = -1
        self.max = 1
        self.constants = []

    def ajustar(self, min=-1, max=1):
        """
        Función que obtiene los límites del rango.

        :param min: límite inferior del rango.
        :param max: límite superior del rango.
        """

        if type(min) != int and type(min) != float:
            raise TypeError("La variable min debe ser numérica.")
        if type(max) != int and type(max) != float:
            raise TypeError("La variable max debe ser numérica")

        if max == min:
            raise ValueError("Las variables max y min no pueden tener el mismo valor")
        if max < min:
            self.min = max
            self.max = min
        else:
            self.min = min
            self.max = max

    def transformar(self, x):
        """
        Función que escala el iterable x al rango dado a la función ajustar.

        :param x: (list/numpy.array/pandas.DataFrame) iterable unidimensional a escalar.
        :return: (list/numpy.array/pandas.DataFrame) iterable unidimensional con los valores escalados.
        """

        # es más fácil trabajar con numpy arrays así que convertimos los datos si es necesario
        if type(x) == list:
            originalData = np.array(x)
        elif type(x).__module__ == "numpy":
            originalData = x
        elif type(x).__module__ == "pandas.core.frame":
            originalData = np.array(x)
        else:
            raise TypeError("Estructura de datos no reconocida.")

        self.constants = []  # guardamos las partes constantes de la fórmula que nos da el escalado de datos

        nrows = len(originalData)
        assert nrows > 1, "No se pueden normalizar los datos si solo hay una muestra"
        ncols = len(originalData[0])
        data = np.zeros((nrows, ncols))  # creamos un array vacío de las mismas dimensiones

        for i in range(ncols):
            col = [originalData[j][i] for j in range(nrows)]
            xmax = max(col)
            xmin = min(col)
            try:
                constants = [self.max - self.min, xmax - xmin, (self.min * xmax - self.max * xmin) / (xmax - xmin)]
                self.constants.append(constants)
                data[:, i] = [round((constants[0] * i2) / constants[1] + constants[2], 5) for i2 in col]

            except:
                raise ValueError("Ha ocurrido un error al transformar los datos.")

        if type(x) == list:
            return data.tolist()
        if type(x).__module__ == "numpy":
            return np.array(data)
        if type(x).__module__ == "pandas.core.frame":
            return pd.DataFrame(data)

    def transformar_inv(self, x):
        """
        Función que invierte el escalado de datos usando el rango pasado a ajustar y las constantes obtenidas de transformar.

        :param x: (list/numpy.array/pandas.DataFrame) iterable unidimensional con valores escalados.
        :return: (list/numpy.array/pandas.DataFrame) iterable unidimensional con los valores desescalados.
        """

        if type(x) == list:
            scaledData = np.array(x)
        elif type(x).__module__ == "numpy":
            scaledData = x
        elif type(x).__module__ == "pandas.core.frame":
            scaledData = np.array(x)
        else:
            raise TypeError("Estructura de datos no reconocida.")

        nrows = len(scaledData)
        ncols = len(scaledData[0])
        data = np.zeros((nrows, ncols))

        for i in range(ncols):
            col = [scaledData[j][i] for j in range(nrows)]

            if max(col) > self.max or min(col) < self.min:
                raise ValueError("Los datos están fuera del rango de normalización.")

            try:
                data[:, i] = [round(((i2 - self.constants[i][2]) * self.constants[i][1]) / self.constants[i][0], 5) for
                              i2 in col]  # calculamos la conversión aplicando la misma fórmula de forma inversa

            except:
                raise ValueError("Ha ocurrido un error al transformar los datos.")

        if type(x) == list:
            return data.tolist()
        if type(x).__module__ == "numpy":
            return data
        if type(x).__module__ == "pandas.core.frame":
            return pd.DataFrame(data)
