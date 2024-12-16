import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import networkx as nx
import pandas as pd
import numpy as np

class Controler:
    def __init__(self, base_datos, modelo_arbol, modelo_reglas):
        self.base_datos = base_datos
        self.modelo_arbol = modelo_arbol
        self.modelo_reglas = modelo_reglas

    def preparar_datos(self, modelo_id, feature_names, class_names):
        """Inserta las características y valores de predicción en la base de datos."""
        for feature in feature_names:
            self.base_datos.insertar_caracteristica(modelo_id, feature)

        for clase in class_names:
            self.base_datos.insertar_valor_prediccion(modelo_id, clase)

    def preprocesar_datos(self, X, modelo_id, feature_names):
        """Convierte características categóricas en numéricas si es necesario."""
        # Convertir X a un DataFrame de pandas si es un ndarray de numpy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_names)  # Asegúrate de que self.feature_names contenga los nombres correctos

        # Verificar si hay columnas categóricas (tipo 'object')
        if X.select_dtypes(include=['object']).empty:
            return X.values, feature_names  # Devolver X como ndarray y los nombres originales si no hay datos categóricos

        # Si hay columnas categóricas, aplicar One-Hot Encoding solo a esas columnas
        transformer = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), X.select_dtypes(include=['object']).columns)],
            remainder='passthrough'
        )
        
        # Ajustar y transformar X
        X_transformed = transformer.fit_transform(X)
        
        # Obtener los nuevos nombres de las características
        new_feature_names = (
            transformer.named_transformers_['cat']
            .get_feature_names_out(X.select_dtypes(include=['object']).columns)
            .tolist()
        )
        
        # Conservar los nombres de las características no categóricas
        non_categorical_feature_names = X.select_dtypes(exclude=['object']).columns.tolist()
        
        # Unir todos los nombres de características
        updated_feature_names = new_feature_names + non_categorical_feature_names

        # Eliminar características existentes pertenecientes al modelo en la base de datos antes de insertar las nuevas
        self.base_datos.eliminar_todas_caracteristicas(modelo_id)
    
        # Insertar las nuevas características en la base de datos
        for feature in updated_feature_names:
          self.base_datos.insertar_caracteristica(modelo_id, feature)

        return X_transformed, updated_feature_names

    def get_parent_id(self, nodo_id, nodos):
        """Determina el ID del nodo padre de un nodo dado."""
        for i, nodo in enumerate(nodos):
            if nodo['left_child'] == nodo_id or nodo['right_child'] == nodo_id:
                return i  # Devuelve el índice del nodo actual como el padre
        return None  # Devuelve None si no tiene padre (caso raíz)

    def entrenar_y_almacenar_arbol(self, X, y, modelo_id, feature_names, class_names):
        """Entrenar y almacenar el modelo de árbol de decisión."""
        # Preprocesar los datos
        X_transformed, updated_feature_names = self.preprocesar_datos(X, modelo_id, feature_names)

        # Entrenar el modelo de árbol de decisión
        self.modelo_arbol.entrenar(X_transformed, y, updated_feature_names)
        nodos, valores = self.modelo_arbol.obtener_nodos()

        # Mapear características y valores de predicción a sus IDs en la base de datos
        caracteristica_ids = {nombre: self.base_datos.obtener_caracteristica_id(modelo_id, nombre) for nombre in updated_feature_names}
        prediccion_ids = {clase: self.base_datos.obtener_valor_prediccion_id(modelo_id, clase) for clase in class_names}

        # Primera etapa: Inserción inicial de nodos
        nodo_id_map = {}
        # Insertar cada nodo del árbol de decisión
        for nodo_id, nodo in enumerate(nodos):
           # Obtener el índice de la característica si es un nodo de decisión
           caracteristica_index = nodo['feature']

           # Asegurarse de que el índice está dentro de los límites
           caracteristica = updated_feature_names[caracteristica_index] if 0 <= caracteristica_index < len(updated_feature_names) else None
           caracteristica_id = caracteristica_ids.get(caracteristica) if caracteristica else None

           # Establecer umbral
           umbral = nodo['threshold'] if caracteristica_id is not None else None
           es_hoja = nodo['left_child'] == -1 and nodo['right_child'] == -1

           # Obtener el ID del valor de predicción si es hoja
           valor_prediccion_id = None
           if es_hoja:
               valor_prediccion = class_names[valores[nodo_id].argmax()]
               valor_prediccion_id = prediccion_ids.get(valor_prediccion)
        
           # Insertar el nodo en la tabla arbol_decision con IDs de características y predicción      
           nodo_db_id = self.base_datos.insertar_nodo(
               modelo_id, None, caracteristica_id, umbral, None, None, es_hoja, valor_prediccion_id
            )
           nodo_id_map[nodo_id] = nodo_db_id

        # Segunda etapa: Actualización de relaciones padre-hijo
        for nodo_id, nodo in enumerate(nodos):
           # Obtener el ID del nodo padre
           nodo_padre_id = nodo_id_map.get(self.get_parent_id(nodo_id, nodos))
           # Actualizar los indices de los nodos izquierdo y derecho
           nodo_izquierdo_id = nodo_id_map.get(nodo['left_child']) if nodo['left_child'] != -1 else None
           nodo_derecho_id = nodo_id_map.get(nodo['right_child']) if nodo['right_child'] != -1 else None

           self.base_datos.actualizar_relaciones_nodo(
               nodo_id_map[nodo_id], nodo_padre_id, nodo_izquierdo_id, nodo_derecho_id
            )
           
        # Preparar listas de IDs para características y clases
        feature_ids = [caracteristica_ids[nombre] for nombre in updated_feature_names]
        class_ids = [str(prediccion_ids[clase]) for clase in class_names]

        print(es_hoja)

        self.visualizar_arbol(feature_ids, class_ids)

    def visualizar_arbol(self, feature_ids, class_ids):
        """Visualizar el árbol de decisión entrenado, mostrando los identificadores de las características y valores de predicción."""
        if self.modelo_arbol.modelo is not None:
           # Dibujar el árbol usando los identificadores proporcionados
           plt.figure(figsize=(14, 12))  # Ajusta el tamaño de la figura
           plot_tree(
                self.modelo_arbol.modelo,
                filled=True,
                feature_names=feature_ids,
                class_names=class_ids
            )
           plt.title("Árbol de Decisión")
           plt.show()  # Mostrar la figura solo una vez
        else:
           raise Exception("El modelo no está entrenado")

    def entrenar_y_almacenar_reglas(self, modelo_id, datos, min_support=0.1, min_confidence=0.3):
        """Entrenar y almacenar solo el modelo de reglas de asociación."""
        datos_bool = datos.astype(bool)
        self.modelo_reglas.entrenar(datos_bool, min_support, min_confidence)
        reglas = self.modelo_reglas.obtener_reglas()

        for _, regla in reglas.iterrows():
            antecedente = ', '.join(list(regla['antecedents']))
            consecuente = ', '.join(list(regla['consequents']))
            soporte = regla['support']
            confianza = regla['confidence']
            lift = regla['lift']

            # Insertar en la tabla
            self.base_datos.insertar_regla(modelo_id, antecedente, consecuente, soporte, confianza, lift)

        # Visualizar las reglas de asociación como un gráfico de red
        self.visualizar_reglas_de_asociacion()

    def visualizar_reglas_de_asociacion(self):
        """Visualizar las reglas de asociación como un grafo donde las aristas representan relaciones,
        y las etiquetas muestran el soporte, confianza y lift."""
        reglas = self.modelo_reglas.obtener_reglas()
    
        # Crear el gráfico de red
        G = nx.DiGraph()

        # Añadir nodos y aristas con atributos de soporte, confianza y lift
        for _, regla in reglas.iterrows():
            antecedente = ', '.join(list(regla['antecedents']))
            consecuente = ', '.join(list(regla['consequents']))
            soporte = regla['support']
            confianza = regla['confidence']
            lift = regla['lift']

            # Agregar la arista con atributos de soporte, confianza y lift
            G.add_edge(antecedente, consecuente, support=soporte, confidence=confianza, lift=lift)

        # Ajustar posiciones con una distancia reducida entre nodos
        pos = nx.spring_layout(G, k=1.5, seed=49)

        # Dibujar nodos y etiquetas
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Dibujar las aristas como relaciones simples
        nx.draw_networkx_edges(
            G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=38,
            edge_color='black', width=1, connectionstyle="arc3,rad=0.32"
        )
    
        # Agregar etiquetas de soporte, confianza y lift sobre las aristas
        edge_labels = {
            (u, v): f"Sup: {d['support']:.2f}, Conf: {d['confidence']:.2f}, Lift: {d['lift']:.2f}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

        # Leyenda
        plt.title("Grafo para Reglas de Asociación")
        plt.show()