from Controler import Controler
from Database import Database
from DecisionTreeModel import DecisionTreeModel
from AssociationRulesModel import AssociationRulesModel
from sklearn.datasets import load_iris
import pandas as pd

# Configuración de modo de ejecución: Cambia el valor de 'modo' para ejecutar solo el árbol, solo reglas o ambos
modo = "reglas"  # "arbol", "reglas" o "ambos"

# Inicializar base de datos  y modelos
db = Database(database="prueba", user="postgres", password="RealMadridVenom")
modelo_arbol = DecisionTreeModel()
modelo_reglas = AssociationRulesModel()
controlador = Controler(db, modelo_arbol, modelo_reglas)
# Lógica de creación del modelo
db.crear_tabla_modelo()
modelo_id = db.insertar_modelo("Cuarto Modelo", "Se usa el dataset de encuestas para modelo de arboles de decisión", "Daniel")

if modo == "arbol" or modo == "ambos":
    # Crear tablas en la base de datos
    db.crear_tablas()
    # Seleccionar el conjunto de datos a usar
    usar_iris = False  # Cambiar a "False" si deseas usar el conjunto de datos de coches de lo contrario "True" para el dataset iris
    usar_transacciones = False  # Cambiar a "True" si deseas usar el conjunto de datos de transacciones de productos
    usar_encuestas = True  # Cambiar a True para usar el dataset de encuestas_hoteles
    usar_hoteles = False  # Cambiar a True para usar el dataset de hoteles 

    if usar_iris:
        # Usar el dataset Iris
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names  # Obtener los nombres de las características
        class_names = iris.target_names  # Obtener los nombres de las clases

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
        X = datos_encuestas.drop(columns=["Q00001", "Q00002", "Q00003", "Q00004", "FrontDesk01[G02Q06]", "FrontDesk01[G02Q07]", "Room01[G03Q09]", "Room01[G03Q10]", 
                                        "Restaurant01[G05Q15]", "Restaurant01[G05Q16]", "Bar01[G06Q21]", "Bar01[G06Q22]", "Personal01[G06Q002]", "Personal01[G06Q003]",
                                        "Outdoor01[G08Q25]", "Outdoor01[G08Q24]", "Animation01[G0003]", "Animation01[G0004]", "Animation01[G0005]", "Pool01[G10Q30]",
                                        "Pool01[G10Q31]", "Suggest01", "idpolo", "dia_del_mes"], axis=1)  # Ajustar según lógica de negocio
        y = datos_encuestas['Suggest01']  # La variable objetivo

        # Nombres de las características y clases
        feature_names = X.columns.tolist()  # Nombres de las características

        # Obtener los valores únicos de la variable objetivo
        valores_objetivo = y.unique()

        # Mapear los valores únicos a sus descripciones
        mapping_valores = {0: 'Recomendado', 1: 'No Recomendado'}

        # Generar class_names a partir de los valores únicos y el mapeo
        class_names = [mapping_valores[val] for val in sorted(valores_objetivo)]

    elif usar_hoteles:
        # Usar el conjunto de datos de encuestas_hoteles desde la base de datos
        datos_hoteles = db.obtener_datos_hoteles()

        # Separar características y objetivo
        X = datos_hoteles.drop(columns=["val_llm", "val_h", "l_comentario", "polo", "modality", "segment", "resort", "staff", "cuba", "comida", "playa", "great", "bien", "servicio", "always", "todos",
                                        "amazing", "buffet","excelente"], axis=1) # Ajustar según lógica de negocio
        y = datos_hoteles['val_h']  # La variable objetivo

        # Nombres de las características y clases
        feature_names = X.columns.tolist()  # Nombres de las características
        
        # Obtener los valores únicos de la variable objetivo
        valores_objetivo = y.unique()

        # Mapear los valores únicos a sus descripciones
        mapping_valores = {-1: 'Valoración Negativa', 0: 'Valoración Neutra', 1: 'Valoración Positiva'}
    
        # Generar class_names a partir de los valores únicos y el mapeo
        class_names = [mapping_valores[val] for val in sorted(valores_objetivo)]

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
    controlador.entrenar_y_almacenar_arbol(X, y, modelo_id, feature_names, class_names)  # Pasar también los nombres de las clases

if modo == "reglas" or modo == "ambos":
    # Crear tablas en la base de datos
    db.crear_tabla_reglas()
    # Seleccionar el conjunto de datos a usar
    usar_transacciones = False  # Cambiar a "True" si deseas usar el dataset de transacciones de prendas de lo contrario "False" para el dataset de transacciones de productos
    usar_encuestas = True  # Cambiar a True para usar el dataset de encuestas_hoteles
    usar_hoteles = False  # Cambiar a True para usar el dataset de hoteles 

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
        datos_sin_columna_sobrante = datos_encuestas.drop(columns=["Q00001", "Q00002", "Q00003", "Q00004", "FrontDesk01[G02Q06]", "FrontDesk01[G02Q07]", "Room01[G03Q09]", "Room01[G03Q10]", 
                                                                   "Restaurant01[G05Q15]", "Restaurant01[G05Q16]", "Bar01[G06Q21]", "Bar01[G06Q22]", "Personal01[G06Q002]", "Personal01[G06Q003]",
                                                                   "Outdoor01[G08Q25]", "Outdoor01[G08Q24]", "Animation01[G0003]", "Animation01[G0004]", "Animation01[G0005]", "Pool01[G10Q30]",
                                                                   "Pool01[G10Q31]", "Suggest01", "idpolo", "dia_de_semana", "mes", "dia_del_mes"], axis=1) # Ajustar según lógica de negocio 

    elif usar_hoteles:
        # Usar el conjunto de datos de encuestas_hoteles desde la base de datos
        datos_hoteles = db.obtener_datos_hoteles()

        # Ignorar columnas  
        datos_sin_columna_sobrante  = datos_hoteles.drop(columns=["val_llm", "val_h", "l_comentario", "polo", "modality", "segment", "resort", "staff", "cuba", "comida", "playa", "great", "bien", "servicio", "always", "todos",
                                     "amazing", "buffet","excelente"], axis=1) # Ajustar según lógica de negocio  

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
    controlador.entrenar_y_almacenar_reglas(modelo_id, datos_sin_columna_sobrante)

# Cerrar la conexión a la base de datos
db.cerrar_conexion()