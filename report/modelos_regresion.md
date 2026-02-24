# Documentación de Modelos de Regresión — Caso 5

> Sección del notebook `Caso5.ipynb` a partir de **Regresión Lineal con Validación Cruzada**.

---

## 1. Regresión Lineal con Validación Cruzada

### Contexto

Este bloque evalúa el modelo base usando validación cruzada, estrategia requerida dado el tamaño reducido del dataset (768 muestras). Al no separar un conjunto de prueba fijo, la validación cruzada garantiza que todos los datos sean usados tanto para entrenar como para evaluar, reduciendo el sesgo en la estimación del error.

### Librerías utilizadas

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
```

### Pipeline de Regresión Lineal

Se construye un pipeline que encadena el escalado de datos con el modelo de regresión lineal, asegurando que el preprocesamiento sea consistente en cada fold de validación.

```python
lr_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
```

**Por qué usar un Pipeline:**
- Evita data leakage: el escalador aprende los parámetros sólo sobre los datos de entrenamiento en cada fold.
- Garantiza reproducibilidad y consistencia del proceso.

### Aplicación de Validación Cruzada (5-Fold)

```python
cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -cv_scores
```

**Parámetros clave:**

| Parámetro | Valor | Descripción |
|---|---|---|
| `cv` | 5 | Número de particiones (folds) |
| `scoring` | `neg_root_mean_squared_error` | Métrica de evaluación (negativa por convención de sklearn) |

**Funcionamiento de `cross_val_score`:**
Divide los datos en 5 partes iguales. En cada iteración, entrena el modelo con 4 partes y lo evalúa con la restante, repitiendo el proceso 5 veces. Esto evita que los resultados dependan de una división aleatoria particular de los datos.

**Por qué se multiplica por `-1`:**
`sklearn` devuelve métricas de error como valores negativos (mayor es mejor). Multiplicar por `-1` convierte el RMSE a su valor real positivo.

### Resultados

```
RMSE Promedio:              3.1662
Desviación Estándar RMSE:   0.4627
```

**Interpretación:**

- El modelo lineal predice la carga de calefacción (`Heating_Load`) con un error promedio de **±3.17 kWh/m²** aproximadamente.
- La desviación estándar de 0.46 indica cierta variabilidad entre los folds, lo que sugiere que el modelo es razonablemente estable pero no perfecto en todos los subconjuntos.
- `neg_root_mean_squared_error`: Un valor más bajo indica que las predicciones están más cerca de los valores reales de carga de calefacción (Y1).

---

## 2. Comparación con Regresión Ridge y Estabilidad

### Contexto

La Regresión Ridge añade una penalización a los coeficientes del modelo (regularización L2) para evitar que crezcan demasiado. Esto es especialmente útil cuando existen variables correlacionadas entre sí (multicolinealidad), lo que puede inflar los coeficientes de la regresión lineal ordinaria y reducir su estabilidad.

### Librerías utilizadas

```python
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
```

### Pipeline de Regresión Ridge

```python
ridge_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
```

**Parámetro `alpha=1.0`:**
Controla la fuerza de la regularización. Un `alpha` mayor penaliza más los coeficientes grandes, forzándolos hacia cero. Si los coeficientes cambian drásticamente al comparar Ridge con la regresión lineal, es señal de multicolinealidad entre variables.

### Resultado de Validación Cruzada (Ridge)

```python
cv_scores_ridge = cross_val_score(ridge_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
```

```
RMSE Ridge Promedio: 3.1716
```

### Comparación de Resultados entre Modelos

| Modelo | RMSE Promedio (5-Fold CV) |
|---|---|
| Regresión Lineal | 3.1662 |
| Regresión Ridge (α=1.0) | 3.1716 |

**Interpretación:** Ambos modelos producen resultados prácticamente idénticos. La regularización Ridge no mejora significativamente el RMSE en este caso, lo que indica que la multicolinealidad no está afectando gravemente las predicciones del modelo lineal base.

### Comparación de Coeficientes

Se entrenaron ambos modelos sobre el dataset completo para extraer y comparar los coeficientes de cada característica:

```python
lr_pipeline.fit(X, y)
ridge_pipeline.fit(X, y)

coef_df = pd.DataFrame({
    'Feature': physical_features,
    'Linear': lr_pipeline.named_steps['model'].coef_,
    'Ridge': ridge_pipeline.named_steps['model'].coef_
})
```

#### Tabla de Coeficientes

| Feature | Linear | Ridge |
|---|---|---|
| Relative_Compactness | -6.8471 | -5.9228 |
| Surface_Area | -2.8259 | -2.4033 |
| Wall_Area | 1.6336 | 1.6055 |
| Roof_Area | -3.5446 | -3.1189 |
| Overall_Height | 7.2974 | 7.5112 |
| Orientation | -0.0261 | -0.0261 |
| Glazing_Area | 2.6537 | 2.6502 |
| Glazing_Area_Distribution | 0.3158 | 0.3162 |
| Overall_Surface | -1.6813 | -1.3406 |

#### Análisis de los Coeficientes

**Variables con mayor impacto en la carga de calefacción:**

1. **`Overall_Height` (+7.30 / +7.51):** Es la variable con mayor influencia positiva. A mayor altura del edificio, mayor es la carga de calefacción predicha. Ridge incluso incrementa ligeramente este coeficiente.

2. **`Relative_Compactness` (-6.85 / -5.92):** Mayor compacidad (edificios más "cúbicos") reduce la carga de calefacción. Ridge reduce la magnitud de este coeficiente, absorbiendo parte de la multicolinealidad con `Surface_Area`.

3. **`Roof_Area` (-3.54 / -3.12):** Un área de techo mayor se asocia a menor carga de calefacción.

4. **`Glazing_Area` (+2.65 / +2.65):** Más superficie de vidrio aumenta la carga de calefacción. El coeficiente es muy estable entre ambos modelos.

**Variables con impacto marginal:**

- **`Orientation` (-0.026):** La orientación del edificio tiene un efecto casi nulo sobre la carga de calefacción, tanto en el modelo lineal como en Ridge.

**`coef_df`:** Esta tabla permite identificar qué características físicas —como `Overall_Height` y la variable construida `Overall_Surface`— tienen mayor peso en el consumo energético del edificio, orientando decisiones de diseño arquitectónico sostenible.

---

## Resumen Comparativo Final

| Aspecto | Regresión Lineal | Regresión Ridge |
|---|---|---|
| RMSE Promedio (5-Fold) | 3.1662 | 3.1716 |
| Regularización | Ninguna | L2 (α=1.0) |
| Sensibilidad a multicolinealidad | Alta | Baja |
| Coeficientes estables | Moderada | Alta |
| Interpretabilidad | Alta | Alta |

**Conclusión:** La Regresión Lineal con validación cruzada establece una línea base sólida con RMSE de 3.17. Ridge no mejora el RMSE de forma significativa, lo que indica que la colinealidad entre variables no degrada materialmente la regresión lineal en este dataset. La variable `Overall_Height` emerge como el predictor dominante de la carga de calefacción.
