from mlxtend.frequent_patterns import apriori, association_rules

class AssociationRulesModel:
    def __init__(self):
        self.reglas = None

    def entrenar(self, datos, min_support=0.1, min_confidence=0.3):
        """Entrena el modelo de reglas de asociación."""
        itemsets_frecuentes = apriori(datos, min_support=min_support, use_colnames=True)
        self.reglas = association_rules(itemsets_frecuentes, metric="confidence", min_threshold=min_confidence)

    def obtener_reglas(self):
        """Obtener reglas con soporte, confianza y lift."""
        if self.reglas is not None:
            return self.reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        else:
            raise Exception("El modelo no está entrenado")