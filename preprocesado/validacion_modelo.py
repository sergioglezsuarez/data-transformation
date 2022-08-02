import numpy as np
import pandas as pd


def divide_entrenamiento_test(*datos, tam_train, semilla=None, mezclar=True, balancear=None):
    """Divide los conjuntos de datos pasados como argumentos en subconjuntos de entrenamiento y test.

    :param datos: (list/numpy.array/pandas.DataFrame) listas, numpy arrays o pandas dataframe que representan los conjuntos de datos.
    :param tam_train: (float/int) número o proporción que indica el tamaño de cada conjunto de entrenamiento.
    :param semilla: (int) semilla usada para obtener resultados reproducibles.
    :param mezclar: (bool) booleano que indica si se mezclan los datos antes de hacer la división.
    :param balancear: (list/numpy.array/pandas.DataFrame) iterable que indica a qué clase pertenece cada muestra para
    hacer un balance de muestras en función de estas clases. Si es distinto de None, mezclar debe ser True.
    :return: (list) devuelve una lista con los subconjuntos de entrenamiento y test de cada iterable.
    """

    assert all([len(i) > 0 and len(i2) > 0 for i in datos for i2 in i]), "Hay iterables vacíos."
    assert tam_train > 0 and ((tam_train < 1 and type(tam_train) == float) or
            (tam_train == 1 and (type(tam_train) == int or type(tam_train) == float)) or
            (tam_train > 1 and type(tam_train) == int)), "No se ha pasado un tamaño de set de entrenamiento válido."
    assert type(mezclar) == bool, "La variable mezclar debe ser de tipo booleano."
    assert (balancear is not None and mezclar) or balancear is None, "Si balancear es diferente de None, mezclar debe ser True."
    assert balancear is None or (balancear is not None and len(balancear) == len( datos[0])), "Si balancear es diferente de None, debe tener la misma longitud que todos los iterables"
    assert semilla is None or (semilla is not None and mezclar), "Si semilla es distinto a None, mezclar debe ser True"

    results = []  # en results se almacenan los subconjuntos de entrenamiento y test

    for i in datos:
        assert type(i) == list or type(i).__module__ == "numpy" or type(i).__module__ == "pandas.core.frame", "Los iterables deben ser de tipo lista, numpy array o pandas dataframe."

        # es más fácil trabajar con listas así que convertimos los datos si es necesario
        if type(i).__module__ == "numpy":
            data = i.tolist()
        elif type(i).__module__ == "pandas.core.frame":
            data = i.values.tolist()
        else:
            data = i

        if tam_train == 1:
            if mezclar:
                if semilla is not None:
                    if type(semilla) == int:
                        np.random.seed(semilla)
                    else:
                        raise AssertionError("La semilla debe ser un número entero o None.")
                train = [data[np.random.randint(0, len(data) - 1)]]  # obtenemos una muestra aleatoria
                data.remove(train[0])  # eliminamos la muestra del conjunto de datos
                test = data  # asignamos el subconjunto de test a las muestras restantes del conjunto de datos
            else:
                train = [data[0]]  # obtenemos la primera muestra
                test = data[1:]  # asignamos el resto del conjunto de datos a test

            if type(i).__module__ == "numpy":
                train = np.array(train)
                test = np.array(test)
            if type(i).__module__ == "pandas.core.frame":
                train = pd.DataFrame(train)
                test = pd.DataFrame(test)

            results.append(train)
            results.append(test)
            continue

        if tam_train <= 1:
            tam_train = round(tam_train * len(data))  # si tam_train es una proporción, lo convertimos a número de muestras

        if balancear is not None:
            assert type(balancear) == list or type(balancear).__module__ == "pandas.core.frame" or type(
                balancear).__module__ == "numpy", "La variable balancear debe ser None, una lista, un array numpy o un pandas dataframe."

            # es más fácil trabajar con listas así que convertimos los datos si es necesario
            if type(balancear).__module__ == "numpy":
                balancear = balancear.tolist()
            if type(balancear).__module__ == "pandas.core.frame":
                balancear = balancear[0].values.tolist()

            class_dict = dict()  # diccionario con el número de muestras por clase a meter en el conjunto de entrenamiento
            count_dict = dict()  # diccionario con el número de muestras por clase que hay en el conjunto de entrenamiento

            for klass in balancear:
                if klass not in class_dict:
                    class_dict[klass] = round(tam_train * (balancear.count(klass) / len(
                        balancear)))  # calculamos el número de muestras que hay que meter
                    count_dict[klass] = 0  # inicializamos a 0 el número de muestras por clase

            if sum(class_dict.values())<tam_train:
                class_dict[max(class_dict, key=class_dict.get)] += 1

            train = []

            if semilla is not None:
                if type(semilla) == int:
                    np.random.seed(semilla)
                else:
                    raise AssertionError("La semilla debe ser un número entero o None.")
            p = np.random.permutation(
                len(data))  # aplicamos la misma permutación a la lista de muestras y la lista de etiquetas para mantener el orden
            data = np.array(data)[p].tolist()
            balancear = np.array(balancear)[p].tolist()

            n = len(balancear)

            # mientras el conjunto de entrenamiento no haya llegado al tamaño deseado, se recorre cada clase para ver si hay que añadir una muestra de esa clase
            while len(train) < tam_train:
                for klass in class_dict:
                    if count_dict[klass] < class_dict[klass]:
                        for i2 in range(n):  # si hay que añadir una muestra de la clase, se recorre el conjunto de datos buscando una muestra de esa clase
                            if balancear[i2] == klass:
                                train.append(data[i2])  # se añade la muestra al conjunto de entrenamiento
                                data.remove(data[i2])  # se elimina del conjunto de datos
                                balancear.remove(klass)  # eliminamos la etiqueta correspondiente a la muestra
                                count_dict[klass] += 1  # actualizamos el diccionario
                                break
                    if len(train) >= tam_train:
                        break

            test = data

            if type(i).__module__ == "numpy":
                train = np.array(train)
                test = np.array(test)
            if type(i).__module__ == "pandas.core.frame":
                train = pd.DataFrame(train)
                test = pd.DataFrame(test)

        else:
            if mezclar:
                if semilla is not None:
                    if type(semilla) == int:
                        np.random.seed(semilla)
                        np.random.shuffle(data)  # mezclamos el conjunto de datos aplicando la semilla
                    else:
                        raise AssertionError("La semilla debe ser un número entero o None.")
                else:
                    np.random.shuffle(data)

            train = data[:tam_train]
            test = data[tam_train:]

            if type(i).__module__ == "numpy":
                train = np.array(train)
                test = np.array(test)
            if type(i).__module__ == "pandas.core.frame":
                train = pd.DataFrame(train)
                test = pd.DataFrame(test)

        results.append(train)
        results.append(test)

    return results
