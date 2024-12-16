-- Tabla para indexar modelos de ML
CREATE TABLE public.grafana_ml_model_index (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL UNIQUE,
    descripcion VARCHAR(255) NOT NULL,
    creador VARCHAR(255) NOT NULL
);

-- Tabla para características asociadas a modelos de árboles de decisión
CREATE TABLE public.caracteristicas (
    modelo_id INT NOT NULL,
    caracteristica_id SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    CONSTRAINT caracteristicas_modelo_nombre_uk UNIQUE (modelo_id, nombre),
    CONSTRAINT fk_caracteristicas_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
);

-- Tabla para valores de predicción asociados a modelos de árboles de decisión
CREATE TABLE public.valores_prediccion (
    modelo_id INT NOT NULL,
    prediccion_id SERIAL PRIMARY KEY,
    nombre_clase VARCHAR(255) NOT NULL,
    CONSTRAINT valores_prediccion_modelo_clase_uk UNIQUE (modelo_id, nombre_clase),
    CONSTRAINT fk_valores_prediccion_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
);

-- Tabla para almacenar árboles de decisión
CREATE TABLE public.grafana_ml_model_arbol_decision (
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

-- Tabla para almacenar reglas de asociación
CREATE TABLE public.grafana_ml_model_reglas_asociacion (
    modelo_id INT NOT NULL,
    id SERIAL PRIMARY KEY,
    antecedente TEXT,
    consecuente TEXT,
    soporte FLOAT,
    confianza FLOAT,
    lift FLOAT,
    CONSTRAINT fk_reglas_modelo FOREIGN KEY (modelo_id) REFERENCES grafana_ml_model_index (id) ON DELETE CASCADE
);
