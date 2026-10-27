# Proyecto IA: Brecha Digital y Rendimiento Escolar en el Perú (ENAHO & ESCALE 2025)

## 📌 Descripción del Proyecto
Pipeline integral de Machine Learning (Aprendizaje No Supervisado, Clasificación Supervisada e Inteligencia Artificial Explicable - XAI) fusionando microdatos oficiales del Estado Peruano (**ENAHO 2025 - INEI** y **ESCALE 2025 - MINEDU**) para identificar perfiles de vulnerabilidad tecnológica y predecir el riesgo de brecha digital en estudiantes de educación básica regular (6 a 17 años).

---

## 🗂 Estructura del Repositorio
```text
├── data/
│   ├── raw/                 # Microdatos originales (Padron.dbf, Padlocal.dbf, ZIPs ENAHO)
│   └── processed/           # Dataset maestro curado (dataset_enaho_escale_2025.parquet y .csv)
├── notebooks/
│   └── ENAHO.ipynb          # Cuaderno Jupyter con todas las fases ejecutadas
├── src/
│   ├── saneamiento.py       # Pipeline de ingesta por streaming, saneamiento y cruce relacional
│   └── build_notebook.py    # Generador automatizado del cuaderno de experimentación
├── informe/
│   └── INFORME_CIENTIFICO_PROYECTO_IA_2025.md  # Paper científico (Estilo IEEE/MDPI)
├── requirements.txt         # Dependencias del proyecto
├── .gitignore               # Exclusión de archivos pesados de datos y checkpoints
└── README.md                # Documentación del proyecto
```

---

## 🔬 Fases del Pipeline de IA

### 1. Auditoría Forense y Saneamiento de Datos Rotos (Fase 0 y 1)
* **Detección de Archivos Falsos**: `04_Docente_01.zip` y `01_Matricula_01.zip` son respuestas de error HTML (404/bloqueo de MINEDU) en lugar de archivos comprimidos. Se reemplazaron utilizando los microdatos íntegros de `Padron.dbf` (137 MB).
* **Solución a las 3 Trampas de INEI**:
  1. *Pérdida de ceros a la izquierda*: Forzado de `dtype=str` con `zfill()` en `UBIGEO` y `CONGLOME`.
  2. *Espacios en blanco*: Conversión de `'   '` a `NaN`.
  3. *Encoding*: Tratamiento de codificación `latin1` para encabezados con tildes y eñes (`AÑO`).
* **Optimización por Streaming**: Extracción de la cohorte 2025 a partir de archivos longitudinales panel con más de 2,600 columnas sin desbordar la memoria RAM.

### 2. Aprendizaje No Supervisado (Fase 2)
* **Algoritmo**: K-Means ($k=3$), evaluado con Coeficiente de Silueta, Índice Davies-Bouldin y Calinski-Harabasz.
* **Reducción de Dimensionalidad**: PCA 2D para proyección visual de los grupos.
* **Perfiles Identificados**:
  * **Cluster 0**: *Urbano Favorecido / Alta Conectividad*.
  * **Cluster 1**: *Rural Altoandino / Aislamiento Digital*.
  * **Cluster 2**: *Urbano Vulnerable / Brecha Moderada*.

### 3. Aprendizaje Supervisado con Cero Data Leakage (Fase 3)
* **Protocolo Anti-Fuga de Datos**: Partición agrupada por conglomerado geográfico (`GroupKFold` con `groups=CONGLOME`).
* **Comparativa de Modelos**:
  | Modelo | ROC-AUC Promedio | F1-Score Macro |
  | :--- | :---: | :---: |
  | **LightGBM** | **0.7967** | **0.7145** |
  | **CatBoost** | 0.7951 | 0.7120 |
  | **XGBoost** | 0.7924 | 0.7088 |
  | **Random Forest** | 0.7852 | 0.7011 |
  | **Regresión Logística (Baseline)** | 0.7410 | 0.6580 |

### 4. Inteligencia Artificial Explicable - XAI con SHAP (Fase 4)
* **SHAP Summary Plot (Beeswarm)**: Identificación de las variables más influyentes (Ingreso neto, condición rural, lengua materna originaria).
* **SHAP Dependence Plot**: Evidencia del impacto cruzado entre lengua materna (Quechua/Aimara) y entorno rural.
* **SHAP Waterfall Plot**: Auditoría local caso por caso (estudiante vulnerable vs. estudiante favorecido).

---

## 🚀 Instalación y Ejecución

```bash
# 1. Clonar el repositorio
git clone <URL_DE_TU_REPOSITORIO>
cd U2_Examen

# 2. Crear y activar entorno virtual
python -m venv env
.\env\Scripts\activate

# 3. Instalar librerías requeridas
python -m pip install -r requirements.txt

# 4. Iniciar Jupyter Lab
jupyter lab
```
Abre el archivo `ENAHO.ipynb` en JupyterLab para reproducir todas las visualizaciones y modelos.
