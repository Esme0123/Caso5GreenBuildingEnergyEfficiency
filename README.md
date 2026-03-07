# Predicción de Eficiencia Energética en Edificios Sustentables 🌿🏢

Este proyecto aplica la metodología **CRISP-DM** para predecir la carga térmica (calefacción) de edificios en función de sus características arquitectónicas, actuando desde el rol de una Consultora LEED.

## 📌 Objetivos del Proyecto
1. **Regresión:** Predecir de manera precisa la variable *Heating Load* utilizando modelos de Machine Learning.
2. **Clasificación:** Categorizar edificios como "Eficientes" (Carga < 14) para toma de decisiones rápidas en la fase de diseño.

## 🛠️ Tecnologías y Herramientas
* **Lenguaje:** Python 3.10+
* **Librerías Clave:** `scikit-learn`, `pandas`, `seaborn`, `matplotlib`
* **Metodología:** CRISP-DM, Pipelines de preprocesamiento, Validación Cruzada (K-Fold).

## 📊 Modelos Evaluados
Se compararon modelos base frente a ensambles avanzados:
* **Regresión:** Regresión Lineal Múltiple, Regresión Ridge (L2) y Random Forest Regressor.
* **Clasificación:** Regresión Logística y Random Forest Classifier.

*Nota: El modelo Random Forest demostró un rendimiento superior, logrando un $R^2$ > 0.99 y un F1-Score casi perfecto, capturando las relaciones no lineales entre la altura del edificio y su demanda térmica.*

## 📁 Estructura del Repositorio
```text
├── data/
│   └── ENB2012_data.xlsx         <- Dataset original
├── notebooks/
│   └── Caso5v.ipynb              <- Notebook con el código completo
├── report/
│   └── Informe_Consultoria.pdf   <- Informe ejecutivo (Business Report)
├── slides/
│   └── Presentacion.pptx         <- Presentación ejecutiva
├── requirements.txt              <- Dependencias del entorno
└── README.md                     <- Portada del proyecto
```
## 🚀 Cómo reproducir este proyecto
* **Clona este repositorio:** git clone https://github.com/Esme0123/Caso5GreenBuildingEnergyEfficiency.git
* **Instala las dependencias:** pip install -r requirements.txt
* **Ejecuta el notebook:** Dentro de la carpeta /notebooks.

## 💡 Hallazgo Principal
La altura total del edificio (Overall_Height) y su superficie de exposición son los factores más críticos para la eficiencia térmica. Las optimizaciones macro-geométricas deben priorizarse antes que las inversiones en materiales de alto costo (como acristalamientos especiales).
