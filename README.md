# Caso 5 — Green Building Energy Efficiency

Análisis de eficiencia energética en edificios sostenibles mediante modelos de Machine Learning. El proyecto predice la **carga de calefacción** (`Heating_Load`) a partir de características físicas del diseño arquitectónico, utilizando el dataset Energy Efficiency del repositorio UCI.

---

## Contexto del Negocio

En la construcción de edificios sostenibles (certificación LEED), predecir el consumo energético **antes de construir** es clave estratégica por tres razones:

- **Ahorro:** Reduce costos en recibos de luz y gas a largo plazo.
- **Equipamiento:** Permite dimensionar correctamente sistemas de aire acondicionado y calefacción.
- **Normativa:** Facilita el cumplimiento de leyes ambientales y metas de reducción de emisiones.

La energía necesaria para climatizar un edificio depende de su forma y diseño: altura, superficie acristalada, orientación, compacidad. Predecir esta carga permite elegir el diseño óptimo y ahorrar dinero.

---

## Objetivo

**Tarea de Regresión:** Predecir el valor numérico de la carga de calefacción (`Y1 = Heating_Load`) a partir de las características físicas del edificio.

> Nota: El dataset también contiene `Y2 = Cooling_Load` (carga de refrigeración), que no es objetivo de este análisis.

---

## Dataset

**Fuente:** [UCI Machine Learning Repository — Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)

**Archivo:** `ENB2012_data.xlsx` (descargado automáticamente desde el notebook)

| Atributo | Descripción |
|---|---|
| Instancias | 768 |
| Variables de entrada | 8 originales + 1 construida |
| Variable objetivo | `Heating_Load` (Y1) |
| Valores faltantes | Ninguno |

### Variables

| Nombre Original | Nombre Renombrado | Descripción |
|---|---|---|
| X1 | `Relative_Compactness` | Compacidad relativa del edificio |
| X2 | `Surface_Area` | Área de superficie total (m²) |
| X3 | `Wall_Area` | Área de paredes (m²) |
| X4 | `Roof_Area` | Área del techo (m²) |
| X5 | `Overall_Height` | Altura total (m) |
| X6 | `Orientation` | Orientación (2=N, 3=E, 4=S, 5=O) |
| X7 | `Glazing_Area` | Área acristalada (proporción) |
| X8 | `Glazing_Area_Distribution` | Distribución del acristalamiento |
| Y1 | `Heating_Load` | **Variable objetivo** — Carga de calefacción (kWh/m²) |
| Y2 | `Cooling_Load` | Carga de refrigeración (no utilizada) |

---

## Estructura del Proyecto

```
Caso-5-Green-Building-Energy-Efficiency/
├── notebooks/
│   └── Caso5.ipynb          # Notebook principal con todo el análisis
├── report/
│   └── modelos_regresion.md # Documentación detallada de los modelos de regresión
├── requirements.txt         # Dependencias del proyecto
└── README.md
```

---

## Pipeline del Análisis

### 1. Importación de Librerías

```python
import pandas as pd
import numpy as np
import requests
import zipfile
import io
```

### 2. Descarga e Inspección del Dataset

Los datos se descargan directamente desde el repositorio UCI para garantizar que todos los usuarios trabajen con exactamente el mismo dataset:

```python
url = "https://archive.ics.uci.edu/static/public/242/energy%2Befficiency.zip"
response = requests.get(url)
```

El archivo ZIP contiene `ENB2012_data.xlsx`, que se lee con pandas. El dataset presenta:
- **768 instancias** sin valores faltantes.
- **8 variables numéricas** de entrada (6 `float64`, 2 `int64`).
- **2 variables objetivo** (`float64`).

### 3. Renombrado de Columnas

Las columnas originales (`X1`–`X8`, `Y1`, `Y2`) se renombran a nombres descriptivos para mejorar la legibilidad del código y los reportes.

### 4. Feature Engineering

Se crea una nueva variable derivada:

```python
df["Overall_Surface"] = df["Wall_Area"] + df["Roof_Area"]
```

**`Overall_Surface`** captura la superficie total de intercambio térmico del edificio (paredes + techo), simplificando la relación con la carga térmica y aportando información contextual al modelo.

### 5. Definición de Features y Target

```python
physical_features = [
    "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    "Overall_Height", "Orientation", "Glazing_Area",
    "Glazing_Area_Distribution", "Overall_Surface"
]
target = "Heating_Load"
```

### 6. Preprocesamiento — Pipeline con StandardScaler

```python
physical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
X_scaled = physical_pipeline.fit_transform(X)
```

El escalado estandariza todas las variables a media=0 y desviación estándar=1, evitando que variables con rangos mayores dominen el aprendizaje del modelo. Se verifica:
- Media de cada columna ≈ 0 (valores del orden de 10⁻¹⁶).
- Desviación estándar de cada columna = 1.0.

---

## Modelos de Machine Learning

### Regresión Lineal con Validación Cruzada

Se evalúa el modelo base con validación cruzada de 5 folds, evitando así dependencia de una división aleatoria particular de los datos.

```python
lr_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
```

**Resultados:**

| Métrica | Valor |
|---|---|
| RMSE Promedio (5-Fold) | **3.1662** |
| Desviación Estándar RMSE | 0.4627 |

El modelo predice la carga de calefacción con un error promedio de ±3.17 kWh/m². La desviación estándar moderada indica estabilidad razonable entre los distintos folds.

### Regresión Ridge (Regularización L2)

Se introduce penalización L2 para controlar el crecimiento de coeficientes y mitigar posibles efectos de multicolinealidad entre variables.

```python
ridge_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
```

**Resultados:**

| Métrica | Valor |
|---|---|
| RMSE Promedio (5-Fold) | **3.1716** |

### Comparación de Modelos

| Modelo | RMSE Promedio | Regularización |
|---|---|---|
| Regresión Lineal | 3.1662 | Ninguna |
| Regresión Ridge (α=1.0) | 3.1716 | L2 |

Ambos modelos producen resultados prácticamente idénticos. La regularización no mejora el RMSE de forma significativa, indicando que la multicolinealidad no afecta gravemente las predicciones en este dataset.

### Análisis de Coeficientes

| Feature | Coef. Lineal | Coef. Ridge |
|---|---|---|
| Relative_Compactness | -6.8471 | -5.9228 |
| Surface_Area | -2.8259 | -2.4033 |
| Wall_Area | 1.6336 | 1.6055 |
| Roof_Area | -3.5446 | -3.1189 |
| **Overall_Height** | **7.2974** | **7.5112** |
| Orientation | -0.0261 | -0.0261 |
| Glazing_Area | 2.6537 | 2.6502 |
| Glazing_Area_Distribution | 0.3158 | 0.3162 |
| Overall_Surface | -1.6813 | -1.3406 |

**Variables más influyentes:**
- **`Overall_Height`** (+7.30): La altura es el predictor más fuerte. Edificios más altos consumen más energía de calefacción.
- **`Relative_Compactness`** (-6.85): Edificios más compactos (forma cúbica) requieren menos calefacción.
- **`Roof_Area`** (-3.54): Mayor área de techo se asocia a menor carga de calefacción.
- **`Glazing_Area`** (+2.65): Más superficie acristalada incrementa el consumo de calefacción.
- **`Orientation`** (-0.026): Tiene impacto prácticamente nulo sobre la carga de calefacción.

---

## Resultados y Conclusiones

1. **Línea base sólida:** La Regresión Lineal con validación cruzada establece un RMSE de 3.17 kWh/m², que sirve como referencia para modelos más complejos.

2. **Multicolinealidad controlada:** Ridge no mejora el RMSE de forma significativa, lo que indica que las variables del dataset no generan problemas graves de colinealidad para la predicción.

3. **Altura dominante:** `Overall_Height` es el predictor más importante de la carga de calefacción, seguido por `Relative_Compactness`. Esto tiene sentido físico: edificios altos tienen mayor superficie expuesta al exterior y mayor demanda energética.

4. **Feature engineering útil:** La variable `Overall_Surface` (Wall_Area + Roof_Area) consolida la información de intercambio térmico y es consistente entre los dos modelos evaluados.

5. **Orientación irrelevante:** La variable `Orientation` presenta un coeficiente cercano a cero en ambos modelos, lo que sugiere que —en este dataset— la orientación del edificio no determina significativamente su consumo energético de calefacción.

---

## Documentación Adicional

- [Reporte de Modelos de Regresión](report/modelos_regresion.md) — Documentación detallada de los procedimientos de Regresión Lineal y Ridge con validación cruzada.

---

## Dependencias

Ver `requirements.txt` para la lista completa. Principales librerías:

- `pandas`, `numpy` — Manipulación de datos
- `scikit-learn` — Modelos de ML, pipelines y validación cruzada
- `matplotlib` — Visualización
- `requests`, `openpyxl` — Descarga y lectura del dataset

---

## Uso

1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar el notebook: `jupyter notebook notebooks/Caso5.ipynb`

El notebook descarga el dataset automáticamente desde UCI al ejecutarse; no se requiere ningún archivo de datos local.
