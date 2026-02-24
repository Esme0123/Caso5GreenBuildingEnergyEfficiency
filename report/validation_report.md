# Reporte de Validación Cruzada y Comparación de Modelos

## Introducción
Este documento detalla el procedimiento realizado para evaluar y comparar la capacidad predictiva de los modelos de regresión regularizada **Ridge** y **Lasso** en el contexto de la eficiencia energética de edificios. El objetivo principal es predecir la carga de calefacción (`Heating_Load`) basándose en características físicas del diseño arquitectónico.

## Metodología

### 1. Preparación de los Datos
Se utilizaron los datos previamente procesados, los cuales incluyen:
- **Ingeniería de variables:** Creación de `Overall_Surface` (suma de `Wall_Area` y `Roof_Area`).
- **Escalado:** Aplicación de `StandardScaler` a todas las variables físicas para asegurar que la regularización actúe de manera equitativa sobre todos los coeficientes.

### 2. Modelos Evaluados
Se seleccionaron dos variantes de regresión lineal con regularización:
- **Ridge (L2):** Añade una penalización proporcional al cuadrado de los coeficientes. Es útil cuando se espera que todas las variables tengan una pequeña contribución al modelo.
- **Lasso (L1):** Añade una penalización proporcional al valor absoluto de los coeficientes. Tiene la propiedad de reducir algunos coeficientes a cero, realizando una selección automática de variables.

### 3. Procedimiento de Validación Cruzada
Para obtener una evaluación imparcial y robusta del rendimiento de los modelos, se implementó:
- **K-Fold Cross-Validation (K=10):** El conjunto de datos se dividió en 10 partes iguales. En cada iteración, se entrenó el modelo en 9 partes y se evaluó en la restante.
- **Métrica de Evaluación:** Se utilizó el **RMSE (Root Mean Squared Error)**. Esta métrica es ideal ya que penaliza los errores grandes y mantiene las unidades originales de la variable objetivo (energía).
- **Consistencia:** Se utilizó una semilla aleatoria (`random_state=42`) para asegurar que la comparación sea justa y reproducible.

## Resultados Esperados
El procedimiento en el notebook genera un reporte de la forma:
- **Ridge RMSE:** Promedio ± Desviación Estándar.
- **Lasso RMSE:** Promedio ± Desviación Estándar.

## Justificación
La elección de estos modelos y este método de validación se justifica por:
- **Robustez:** La validación cruzada reduce la probabilidad de que los resultados dependan de una partición específica de los datos.
- **Interpretabilidad:** El uso de RMSE permite entender directamente cuánto se equivoca el modelo en promedio en términos de carga térmica.
- **Optimización:** Comparar Ridge y Lasso permite identificar si la complejidad del problema se beneficia más de una reducción global de coeficientes (Ridge) o de una simplificación del modelo (Lasso).

---
*Este reporte fue generado como parte del análisis de eficiencia energética en el Caso de Estudio 5.*
