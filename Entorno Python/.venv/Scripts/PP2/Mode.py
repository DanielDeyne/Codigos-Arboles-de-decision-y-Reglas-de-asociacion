from Controler import Controler
from Database import Database
from DecisionTreeModel import DecisionTreeModel
from AssociationRulesModel import AssociationRulesModel
from sklearn.datasets import load_iris
import pandas as pd

# Configuración de modo de ejecución: Cambia el valor de 'modo' para ejecutar solo el árbol, solo reglas o ambos
modo = "arbol"  # "arbol", "reglas" o "ambos"

# Inicializar base de datos  y modelos
db = Database(database="prueba", user="postgres", password="RealMadridVenom")
modelo_arbol = DecisionTreeModel()
modelo_reglas = AssociationRulesModel()
controlador = Controler(db, modelo_arbol, modelo_reglas)
# Lógica de creación del modelo
db.crear_tabla_modelo()
modelo_id = db.insertar_modelo("Iris (sepal length (cm) y petal length (cm))", "Se usa el dataset iris para modelo de árboles de decisión sin sepal width (cm) y petal width (cm)", "Daniel")

if modo == "arbol" or modo == "ambos":
    # Crear tablas en la base de datos
    db.crear_tablas()
    # Seleccionar el conjunto de datos a usar
    usar_iris = True  # Cambiar a "False" si deseas usar el conjunto de datos de coches de lo contrario "True" para el dataset iris
    usar_transacciones = False  # Cambiar a "True" si deseas usar el conjunto de datos de transacciones de productos
    usar_encuestas = False  # Cambiar a True para usar el dataset de encuestas_hoteles
    usar_comentarios = False  # Cambiar a True para usar el dataset de comentarios_hoteles 

    # Definir el valor por defecto para max_depth
    max_depth = None

    # Definir el valor por defecto para class_weight
    class_weight = None

    if usar_iris:
        # Usar el dataset Iris
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names  # Obtener los nombres de las características
        class_names = iris.target_names  # Obtener los nombres de las clases

        # Convertir X en un DataFrame de pandas para facilitar el manejo
        X_df = pd.DataFrame(X, columns=feature_names)

        # Lista de columnas a eliminar
        columnas_a_eliminar = ['sepal width (cm)', 'petal width (cm)']  # Cambiar esta lista según lo que se desee eliminar: "sepal length (cm)", "sepal width (cm)",
                                                                                                                           # "petal length (cm)", "petal width (cm)"

        # Eliminar columnas solo si la lista no está vacía
        if columnas_a_eliminar:
            X_df = X_df.drop(columns=columnas_a_eliminar)

        # Actualizar X y feature_names
        X = X_df.values
        feature_names = X_df.columns.tolist()  # Nombres de características actualizados

        # Asignar max_depth específico
        max_depth = 5

        # Asignar class_weight específico
        class_weight = "balanced"

    elif usar_transacciones:
        # Usar el conjunto de datos de transacciones de productos
        datos = pd.DataFrame([
            {'pan': 1, 'leche': 0, 'mantequilla': 1, 'cereal': 0, 'comprador': 'Cliente A'},
            {'pan': 1, 'leche': 1, 'mantequilla': 0, 'cereal': 1, 'comprador': 'Cliente B'},
            {'pan': 0, 'leche': 1, 'mantequilla': 1, 'cereal': 0, 'comprador': 'Cliente C'},
            {'pan': 1, 'leche': 1, 'mantequilla': 1, 'cereal': 0, 'comprador': 'Cliente D'},
            {'pan': 0, 'leche': 0, 'mantequilla': 1, 'cereal': 1, 'comprador': 'Cliente E'},
        ])
        
        # Para árbol de decisión
        X = datos.drop('comprador', axis=1)  # Usar todas las columnas excepto 'comprador' como características
        y = datos['comprador']  # 'comprador' será la variable objetivo

        feature_names = X.columns.tolist()  # Nombres de características
        class_names = datos['comprador'].unique().tolist()  # Clases únicas

    elif usar_encuestas:
        # Usar el conjunto de datos de encuestas_hoteles desde la base de datos
        datos_encuestas = db.obtener_datos_encuestas()

        # Separar características y objetivo
        X = datos_encuestas.drop(columns=["Q00001", "Q00002", "Q00003", "Q00004", "Suggest01", "idpolo", "dia_de_semana", "mes",
                                          "dia_del_mes"], axis=1)  # Ajustar según lógica de negocio
        y = datos_encuestas['Suggest01']  # La variable objetivo

        # Nombres de las características y clases
        feature_names = X.columns.tolist()  # Nombres de las características

        # Obtener los valores únicos de la variable objetivo
        valores_objetivo = y.unique()

        # Mapear los valores únicos a sus descripciones
        mapping_valores = {0: 'Avala', 1: 'No Avala'}

        # Generar class_names a partir de los valores únicos y el mapeo
        class_names = [mapping_valores[val] for val in sorted(valores_objetivo)]

        # Asignar max_depth específico
        max_depth = 3

        # Asignar class_weight específico
        class_weight = "balanced"

    elif usar_comentarios:
        # Usar el conjunto de datos de comentarios_emitidos desde la base de datos
        datos_comentarios = db.obtener_datos_comentarios()

        # Separar características y objetivo
        X = datos_comentarios.drop(columns=["val_h", "polo", "modality", "segment", "dia_de_semana", "dia_del_mes", "val_llm"], axis=1) # Ajustar según lógica de negocio
        y = datos_comentarios['val_llm']  # Cambiar entre "val_h" y "val_llm" para variable objetivo

        # Nombres de las características y clases
        feature_names = X.columns.tolist()  # Nombres de las características
        
        # Obtener los valores únicos de la variable objetivo
        valores_objetivo = y.unique()

        # Mapear los valores únicos a sus descripciones
        mapping_valores = {-1: 'Negativa', 0: 'Neutra', 1: 'Positiva'}
    
        # Generar class_names a partir de los valores únicos y el mapeo
        class_names = [mapping_valores[val] for val in sorted(valores_objetivo)]

        # Asignar max_depth específico
        max_depth = 3

        # Asignar class_weight específico
        class_weight = "balanced"

    else:
        # Usar el conjunto de datos de coches
        data = {
            'Marca': ['Toyota', 'Ford', 'Honda', 'Tesla', 'Nissan'],
            'Modelo': ['Corolla', 'Focus', 'Civic', 'Model 3', 'Leaf'],
            'Color': ['Rojo', 'Azul', 'Negro', 'Blanco', 'Verde'],
            'Tipo de combustible': ['Gasolina', 'Diésel', 'Híbrido', 'Eléctrico', 'Petroleo']
        }
        df_coches = pd.DataFrame(data)
        
        # Separar características y objetivo
        X = df_coches.drop('Marca', axis=1)  # Usar todas las columnas excepto 'Marca' como características
        y = df_coches['Marca']  # 'Marca' es la variable objetivo

        feature_names = X.columns.tolist()  # Obtener los nombres de las características
        class_names = df_coches['Marca'].unique().tolist()  # Obtener los nombres de las clases

    controlador.preparar_datos(modelo_id, feature_names, class_names)
    controlador.entrenar_y_almacenar_arbol(X, y, modelo_id, feature_names, class_names, max_depth=max_depth, class_weight=class_weight)  

if modo == "reglas" or modo == "ambos":
    # Crear tablas en la base de datos
    db.crear_tabla_reglas()
    # Seleccionar el conjunto de datos a usar
    usar_transacciones = False  # Cambiar a "True" si deseas usar el dataset de transacciones de prendas de lo contrario "False" para el dataset de transacciones de productos
    usar_encuestas = False  # Cambiar a True para usar el dataset de encuestas_hoteles
    usar_comentarios = True  # Cambiar a True para usar el dataset de comentarios_hoteles

    # Definir el valor por defecto para filtro_support
    filtro_support=None

    # Definir el valor por defecto para filtro_lift
    filtro_lift=None

    if usar_transacciones:
        datos = pd.DataFrame([
            {'camisa': 1, 'pantalón': 0, 'zapatos': 1, 'sombrero': 0, 'comprador': 'Cliente F'},
            {'camisa': 0, 'pantalón': 1, 'zapatos': 1, 'sombrero': 1, 'comprador': 'Cliente G'},
            {'camisa': 1, 'pantalón': 1, 'zapatos': 0, 'sombrero': 0, 'comprador': 'Cliente H'},
            {'camisa': 0, 'pantalón': 0, 'zapatos': 1, 'sombrero': 1, 'comprador': 'Cliente I'},
            {'camisa': 1, 'pantalón': 1, 'zapatos': 1, 'sombrero': 0, 'comprador': 'Cliente J'},
        ])
        # Ignorar la columna 'comprador'
        datos_sin_columna_sobrante = datos.drop('comprador', axis=1)

    elif usar_encuestas:
        # Usar el conjunto de datos de encuestas_hoteles desde la base de datos
        datos_encuestas = db.obtener_datos_encuestas()

        # Ignorar columnas 
        datos_sin_columna_sobrante = datos_encuestas.drop(columns=["Q00001", "Q00002", "Q00003", "Q00004", "FrontDesk01[G02Q06]","FrontDesk01[G02Q07]", "Room01[G03Q09]", 
                                                                "Room01[G03Q10]", "Restaurant01[G05Q15]", "Restaurant01[G05Q16]","Bar01[G06Q21]", "Bar01[G06Q22]", 
                                                                "Personal01[G06Q002]","Personal01[G06Q003]", "Outdoor01[G08Q25]","Outdoor01[G08Q24]", "Animation01[G0003]",
                                                                "Animation01[G0004]", "Animation01[G0005]", "Pool01[G10Q30]", "Pool01[G10Q31]","Suggest01", "idpolo", 
                                                                "dia_de_semana", "mes", "dia_del_mes"], axis=1)  
        
        # Preprocesar columnas de valoraciones
        columnas_valoracion = ["FrontDesk01[G02Q05]", "Room01[G03Q08]", "Restaurant01[G05Q14]", "Bar01[G06Q20]", "Personal01[G06Q001]", "Outdoor01[G08Q23]",
                               "Animation01[G0901]", "Pool01[G10Q29]"]  # Añade todas las columnas relevantes
        
        for columna in columnas_valoracion:
            datos_sin_columna_sobrante[f"{columna}_Pos"] = (datos_sin_columna_sobrante[columna] == 1).astype(int)
            datos_sin_columna_sobrante[f"{columna}_Neu"] = (datos_sin_columna_sobrante[columna] == 0).astype(int)
            datos_sin_columna_sobrante[f"{columna}_Neg"] = (datos_sin_columna_sobrante[columna] == -1).astype(int)
            datos_sin_columna_sobrante.drop(columns=[columna], inplace=True)
        
        # Asignar filtro_support específico
        filtro_support = 0.1

    elif usar_comentarios:
        # Usar el conjunto de datos de encuestas_hoteles desde la base de datos
        datos_comentarios = db.obtener_datos_comentarios()

        # Ignorar columnas  
        datos_sin_columna_sobrante  = datos_comentarios.drop(columns=["l_comentario", "polo", "modality", "segment", "dia_de_semana", "mes", "dia_del_mes", "val_h", "val_llm"], 
                                                                  axis=1) # Con y sin: "val_h" y "val_llm" 
        
        # Verificar si las columnas de valoración están presentes antes de procesarlas
        columnas_valoracion = ["val_h", "val_llm"]
        if all(col in datos_sin_columna_sobrante.columns for col in columnas_valoracion):
            # Preprocesar columnas de valoraciones
            for columna in columnas_valoracion:
                datos_sin_columna_sobrante[f"{columna}_Pos"] = (datos_sin_columna_sobrante[columna] == 1).astype(int)
                datos_sin_columna_sobrante[f"{columna}_Neu"] = (datos_sin_columna_sobrante[columna] == 0).astype(int)
                datos_sin_columna_sobrante[f"{columna}_Neg"] = (datos_sin_columna_sobrante[columna] == -1).astype(int)
                datos_sin_columna_sobrante.drop(columns=[columna], inplace=True)

        # Asignar filtro_support específico
        filtro_support = 0.1

    else:
        datos = pd.DataFrame([
            {'pan': 1, 'leche': 0, 'mantequilla': 1, 'cereal': 0, 'comprador': 'Cliente A'},
            {'pan': 1, 'leche': 1, 'mantequilla': 0, 'cereal': 1, 'comprador': 'Cliente B'},
            {'pan': 0, 'leche': 1, 'mantequilla': 1, 'cereal': 0, 'comprador': 'Cliente C'},
            {'pan': 1, 'leche': 1, 'mantequilla': 1, 'cereal': 0, 'comprador': 'Cliente D'},
            {'pan': 0, 'leche': 0, 'mantequilla': 1, 'cereal': 1, 'comprador': 'Cliente E'},
        ])
        # Ignorar la columna 'comprador'
        datos_sin_columna_sobrante = datos.drop('comprador', axis=1)

    #db.crear_tabla_reglas()
    controlador.entrenar_y_almacenar_reglas(modelo_id, datos_sin_columna_sobrante, filtro_support=filtro_support, filtro_lift=filtro_lift)

# Cerrar la conexión a la base de datos
db.cerrar_conexion()