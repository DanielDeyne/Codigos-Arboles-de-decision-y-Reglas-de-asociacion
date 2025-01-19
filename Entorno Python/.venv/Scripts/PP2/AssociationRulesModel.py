from mlxtend.frequent_patterns import apriori, association_rules

class AssociationRulesModel:
    def __init__(self):
        self.reglas = None

    def entrenar(self, datos, min_support=0.1, min_confidence=0.7, filtro_support=None, filtro_lift=None):
        """
        Entrena el modelo de reglas de asociación y aplica filtros opcionales.
    
        Parámetros:
           datos (DataFrame): Datos transaccionales.
           min_support (float): Soporte mínimo para el algoritmo Apriori.
           min_confidence (float): Confianza mínima para generar reglas.
           filtro_support (float, opcional): Valor mínimo de soporte para filtrar reglas.
           filtro_lift (float, opcional): Valor mínimo de lift para filtrar reglas.
        """

        # Generar los ítems frecuentes
        itemsets_frecuentes = apriori(datos, min_support=min_support, use_colnames=True)

        # Generar las reglas usando la confianza como métrica principal
        reglas = association_rules(itemsets_frecuentes, metric="confidence", min_threshold=min_confidence)

        # Verificar que las reglas cumplen el filtro de confianza
        reglas = reglas[reglas['confidence'] >= min_confidence]

        # Aplicar filtros opcionales
        if filtro_support is not None:
           reglas = reglas[reglas['support'] >= filtro_support]
        if filtro_lift is not None:
           reglas = reglas[reglas['lift'] >= filtro_lift]

        # Almacenar las reglas finales
        self.reglas = reglas


    def obtener_reglas(self):
        """Obtener reglas con soporte, confianza y lift."""
        if self.reglas is not None:
            return self.reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        else:
            raise Exception("El modelo no está entrenado")