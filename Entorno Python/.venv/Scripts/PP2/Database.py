import pg8000
import pandas as pd

class Database:
    def __init__(self, database, user, password, host='localhost', port=5432):
        """Inicializa la conexión a la base de datos."""
        self.conn = pg8000.connect(database=database, user=user, password=password, host=host, port=port)
        self.cursor = self.conn.cursor()

    def crear_tabla_modelo (self):
        """Crear la tabla para almacenar los modelos."""
         # Tabla para almacenar los modelos
        query_modelo = """
        CREATE TABLE IF NOT EXISTS grafana_ml_model_index (
            id SERIAL PRIMARY KEY,
            nombre VARCHAR(255) NOT NULL UNIQUE,
            descripcion VARCHAR(255) NOT NULL,
            creador VARCHAR(255) NOT NULL
        );
        """

         # Ejecuta las consultas
        self.cursor.execute(query_modelo)
        self.conn.commit()

    def insertar_modelo(self, nombre, descripcion, creador):
        """Inserta un modelo en la tabla 'grafana_ml_model_index' y devuelve su ID."""
        query = """
        INSERT INTO grafana_ml_model_index (nombre, descripcion, creador)
        VALUES (%s, %s, %s)
        ON CONFLICT (nombre) DO NOTHING RETURNING id;
        """
        self.cursor.execute(query, (nombre, descripcion, creador,))
        modelo_id = self.cursor.fetchone()
        
        # Si no se inserta, busca el ID existente
        if modelo_id:
            return modelo_id[0]
        
        # Busca el ID existente
        self.cursor.execute("SELECT id FROM grafana_ml_model_index WHERE nombre = %s;", (nombre,))
        return self.cursor.fetchone()[0]  # Devuelve el ID existente


    def crear_tablas(self):
        """Crear las tablas para almacenar el modelo de árbol de decisión, características y valores de predicción."""
        
        
        # Tabla para almacenar las características
        query_caracteristicas = """
        CREATE TABLE IF NOT EXISTS caracteristicas (
            modelo_id INT NOT NULL,
            caracteristica_id SERIAL PRIMARY KEY,
            nombre VARCHAR(255) NOT NULL,
            CONSTRAINT caracteristicas_modelo_nombre_uk UNIQUE (modelo_id, nombre),
            CONSTRAINT fk_caracteristicas_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
        );
        """
        
        # Tabla para almacenar los valores de predicción (nombre de la clase)
        query_valores_prediccion = """
        CREATE TABLE IF NOT EXISTS valores_prediccion (
            modelo_id INT NOT NULL,
            prediccion_id SERIAL PRIMARY KEY,
            nombre_clase VARCHAR(255) NOT NULL,
            CONSTRAINT valores_prediccion_modelo_clase_uk UNIQUE (modelo_id, nombre_clase),
            CONSTRAINT fk_valores_prediccion_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
        );
        """
        
        # Tabla para el árbol de decisión con relaciones a las otras dos tablas
        query_arbol = """
        CREATE TABLE IF NOT EXISTS grafana_ml_model_arbol_decision (
            modelo_id INT NOT NULL,
            nodo_id SERIAL PRIMARY KEY,
            nodo_padre INT,
            caracteristica INT REFERENCES caracteristicas(caracteristica_id) ON DELETE CASCADE,
            umbral FLOAT,
            nodo_izquierdo INT,
            nodo_derecho INT,
            es_hoja BOOLEAN,
            valor_prediccion INT REFERENCES valores_prediccion(prediccion_id),
            CONSTRAINT fk_arbol_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
        );
        """
        
        # Ejecuta las consultas
        self.cursor.execute(query_caracteristicas)
        self.cursor.execute(query_valores_prediccion)
        self.cursor.execute(query_arbol)
        self.conn.commit()

    def insertar_caracteristica(self, modelo_id, nombre):
        """Inserta una característica en la tabla y devuelve su ID."""
        query = """
        INSERT INTO caracteristicas (modelo_id, nombre)
        VALUES (%s, %s)
        ON CONFLICT (modelo_id, nombre) DO NOTHING RETURNING caracteristica_id;
        """
        self.cursor.execute(query, (modelo_id, nombre,))
        caracteristica_id = self.cursor.fetchone()
        
        # Si no se inserta, busca el ID existente
        if caracteristica_id:
            return caracteristica_id[0]
        
        # Busca el ID existente
        self.cursor.execute("SELECT caracteristica_id FROM caracteristicas WHERE modelo_id = %s AND nombre = %s;", (modelo_id, nombre,))
        return self.cursor.fetchone()[0]  # Devuelve el ID existente
    
    def eliminar_todas_caracteristicas(self, modelo_id):
        """Elimina todas las características asociadas a un modelo específico en la base de datos."""
        if modelo_id is None:
           raise ValueError("El parámetro 'modelo_id' es obligatorio y no puede ser None.")
        
        with self.conn.cursor() as cursor:
            cursor.execute("DELETE FROM caracteristicas WHERE modelo_id = %s;", (modelo_id,))
        self.conn.commit()

    def insertar_valor_prediccion(self, modelo_id, nombre_clase):
        """Inserta un valor de predicción (nombre de la clase) en la tabla y devuelve su ID."""
        query = """
        INSERT INTO valores_prediccion (modelo_id, nombre_clase)
        VALUES (%s, %s)
        ON CONFLICT (modelo_id, nombre_clase) DO NOTHING RETURNING prediccion_id;
        """
        self.cursor.execute(query, (modelo_id, nombre_clase,))
        prediccion_id = self.cursor.fetchone()
        
        # Si no se inserta, busca el ID existente
        if prediccion_id:
            return prediccion_id[0]

        # Busca el ID existente
        self.cursor.execute("SELECT prediccion_id FROM valores_prediccion WHERE modelo_id = %s AND nombre_clase = %s;", (modelo_id, nombre_clase,))
        return self.cursor.fetchone()[0]  # Devuelve el ID existente

    def insertar_nodo(self, modelo_id, nodo_padre, caracteristica, umbral, nodo_izquierdo, nodo_derecho, es_hoja, valor_prediccion):
        """Insertar un nodo en la tabla del árbol de decisión y devolver su ID."""
        query_insert_nodo = """
        INSERT INTO grafana_ml_model_arbol_decision (modelo_id, nodo_padre, caracteristica, umbral, nodo_izquierdo, nodo_derecho, es_hoja, valor_prediccion)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING nodo_id;
        """
        self.cursor.execute(query_insert_nodo, (modelo_id, nodo_padre, caracteristica, umbral, nodo_izquierdo, nodo_derecho, es_hoja, valor_prediccion))
        nodo_id = self.cursor.fetchone()
    
        # Confirma los cambios
        self.conn.commit()

        # Retorna el ID del nodo recién insertado
        return nodo_id[0]

    def actualizar_relaciones_nodo(self, nodo_id, nodo_padre, nodo_izquierdo, nodo_derecho):
        """Actualizar relaciones de un nodo en la tabla."""
        query = """
        UPDATE grafana_ml_model_arbol_decision
        SET nodo_padre = %s, nodo_izquierdo = %s, nodo_derecho = %s
        WHERE nodo_id = %s;
        """
        self.cursor.execute(query, (nodo_padre, nodo_izquierdo, nodo_derecho, nodo_id))
        self.conn.commit()

    def obtener_caracteristica_id(self, modelo_id, nombre):
        """Devuelve el ID de una característica a partir de su nombre y modelo_id."""
        query = "SELECT caracteristica_id FROM caracteristicas WHERE modelo_id = %s AND nombre = %s;"
        self.cursor.execute(query, (modelo_id, nombre,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def obtener_valor_prediccion_id(self, modelo_id, nombre_clase):
        """Devuelve el ID de un valor de predicción a partir de su nombre y modelo_id."""
        query = "SELECT prediccion_id FROM valores_prediccion WHERE modelo_id = %s AND nombre_clase = %s;"
        self.cursor.execute(query, (modelo_id, nombre_clase,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def crear_tabla_reglas(self):
        """Crear la tabla para almacenar las reglas de asociación."""
        query = """
        CREATE TABLE IF NOT EXISTS grafana_ml_model_reglas_asociacion (
            modelo_id INT NOT NULL,
            id SERIAL PRIMARY KEY,
            antecedente TEXT,
            consecuente TEXT,
            soporte FLOAT,
            confianza FLOAT,
            lift FLOAT,
            CONSTRAINT fk_reglas_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
        );
        """
        self.cursor.execute(query)
        self.conn.commit()

    def insertar_regla(self, modelo_id, antecedente, consecuente, soporte, confianza, lift):
        """Insertar una regla de asociación en la tabla grafana_ml_model_reglas_asociacion."""
        query = """
        INSERT INTO grafana_ml_model_reglas_asociacion (modelo_id, antecedente, consecuente, soporte, confianza, lift)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        self.cursor.execute(query, (modelo_id, antecedente, consecuente, soporte, confianza, lift))
        self.conn.commit()

    def obtener_datos_encuestas(self):
        """Obtener los datos de encuestas desde la base de datos y devolverlos como un DataFrame."""
        query = "SELECT * FROM encuestas_hoteles"
        
        # Ejecutar la consulta 
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        # Obtener los nombres de las columnas
        column_names = [desc[0] for desc in self.cursor.description]

        # Crear un DataFrame con los resultados
        df = pd.DataFrame(rows, columns=column_names)

        return df

    def obtener_datos_hoteles(self):
        """Obtener los datos de hoteles desde la base de datos y devolverlos como un DataFrame."""
        query = "SELECT * FROM hoteles"
        
        # Ejecutar la consulta
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        # Obtener los nombres de las columnas
        column_names = [desc[0] for desc in self.cursor.description]

        # Crear un DataFrame con los resultados
        df = pd.DataFrame(rows, columns=column_names)

        return df

    def cerrar_conexion(self):
        """Cerrar la conexión a la base de datos."""
        self.cursor.close()
        self.conn.close()