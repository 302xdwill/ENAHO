# BRECHA DIGITAL Y VULNERABILIDAD ESCOLAR EN EL PERÚ: UN ENFOQUE DE APRENDIZAJE AUTOMÁTICO ROBUSTO Y EXPLICABILIDAD ALGORÍTMICA (XAI) SOBRE MICRODATOS ENAHO Y ESCALE 2025

**Autor(es):** Estudiantes de Ingeniería de Sistemas  
**Institución:** Universidad Peruana Unión (UPeU)  
**Curso:** Inteligencia Artificial  
**Fecha:** Octubre de 2026  

---

### RESUMEN (ABSTRACT)
El acceso desigual a las Tecnologías de la Información y Comunicación (TIC) constituye uno de los principales determinantes de la exclusión socioeducativa en el Perú, acentuado de forma desproporcionada en áreas rurales y poblaciones originarias. El presente artículo desarrolla un pipeline integral, reproducible y libre de fuga de datos (*data leakage*) de Machine Learning, integrando los microdatos de la Encuesta Nacional de Hogares (**ENAHO 2025 - INEI**) con el Padrón de Instituciones Educativas (**ESCALE 2025 - MINEDU**), abarcando una cohorte depurada de **23,668 estudiantes en edad escolar (6 a 17 años)**. 

Se implementó un módulo no supervisado de *Clustering* ($k$-Means con reducción dimensional PCA), validado cuantitativamente mediante el Coeficiente de Silueta, el Índice Davies-Bouldin y el Índice Calinski-Harabasz, descubriendo tres tipologías estructurales de estudiantes. En el modelado supervisado, se evaluaron cinco arquitecturas algorítmicas bajo un protocolo estricto de validación cruzada agrupada por conglomerado geográfico (`GroupKFold`), evitando la contaminación espacial. El algoritmo **LightGBM** demostró el desempeño predictivo superior con un **ROC-AUC de 0.7967** y un **F1-Score Macro de 0.7145**. Finalmente, la auditoría de Inteligencia Artificial Explicable (**SHAP TreeExplainer**) evidenció que el ingreso per cápita del hogar, la condición de ruralidad y la pertenencia a una lengua materna originaria (Quechua/Aimara) interactúan de forma no lineal como barreras sistemáticas de acceso tecnológico, ofreciendo pautas concretas para políticas públicas educativas focalizadas.

**Palabras clave:** *Machine Learning, Brecha Digital, ENAHO 2025, ESCALE, GroupKFold, Zero Data Leakage, LightGBM, SHAP, XAI.*

---

## I. INTRODUCCIÓN
En el contexto peruano pospandemia, la brecha digital ha dejado de ser una simple disparidad de equipamiento para convertirse en un factor condicionante del rezago y la deserción escolar. No obstante, los análisis estadísticos convencionales suelen tratar las variables de forma aislada, ignorando la correlación espacial intrínseca de los hogares que habitan en un mismo conglomerado o distrito.

Este proyecto tiene como objetivo modelar la vulnerabilidad tecnológica de los estudiantes peruanos a través de un enfoque riguroso de Ciencia de Datos e Inteligencia Artificial, abordando:
1. La auditoría y saneamiento de fuentes de datos estatales complejas y parcialmente dañadas.
2. El perfilamiento no supervisado de la población escolar.
3. La predicción de la brecha digital efectiva mediante algoritmos de ensamble (*Gradient Boosting* y *Bagging*) bajo partición espacial libre de fuga.
4. La explicabilidad algorítmica profunda mediante valores de Shapley (*SHAP*).

---

## II. METODOLOGÍA DE DATOS Y ARQUITECTURA DEL PIPELINE

```
+-----------------------------------------------------------------------------------------+
|                                    FUENTES DE DATOS                                     |
|  [ENAHO 2025 - INEI]                                        [ESCALE 2025 - MINEDU]      |
|  - Mod 100: Vivienda/Luz/Internet                           - Padron.dbf: Infraestructura|
|  - Mod 200: Miembros/Edad escolar (6-17)                    - Padlocal.dbf: Georreferencia|
|  - Mod 300: Educación/Lengua/TICs                                                       |
|  - Sumaria: Pobreza/Ingreso neto                                                        |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                  FASE 0 & 1: SANEAMIENTO FORENSE Y STREAMING RELACIONAL                 |
|  - Descarte de archivos HTML de error de MINEDU (04_Docente_01, 01_Matricula_01)         |
|  - Forzado de dtype=str (zfill) en llaves: CONGLOME, VIVIENDA, HOGAR, CODPERSO, UBIGEO  |
|  - Tratamiento de espacios en blanco como NaN y encoding latin1                         |
|  - Extracción en Streaming de cohorte 2025 (_25) desde archivo Panel de 2,600+ columnas |
|  - Agregación distrital de ESCALE (Ratio Alumno/Docente, % Públicos, Coordenadas)       |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                       DATASET MAESTRO INTEGRADO (23,668 ESTUDIANTES)                    |
+-----------------------------------------------------------------------------------------+
                    |                                                 |
                    v                                                 v
+---------------------------------------+   +---------------------------------------------+
|    FASE 2: MODELADO NO SUPERVISADO    |   |     FASE 3: MODELADO SUPERVISADO ROBUSTO    |
|  - Estandarización StandardScaler     |   |  - Partición GroupKFold por CONGLOME        |
|  - Reducción de Dimensión PCA (2D)    |   |  - Modelos: Logistic Regression, RF,        |
|  - k-Means (k=3 validado matemáticam.)|   |    LightGBM, XGBoost, CatBoost              |
|  - Perfilamiento y Bautizo de Grupos  |   |  - Métricas: ROC-AUC, PR-AUC, F1, Matriz CM |
+---------------------------------------+   +---------------------------------------------+
                                                                      |
                                                                      v
                                            +---------------------------------------------+
                                            |     FASE 4: XAI (SHAP TREEEXPLAINER)        |
                                            |  - Beeswarm Global: Top determinantes       |
                                            |  - Scatter/Dependence: Interacción no lineal|
                                            |  - Waterfall: Auditoría individual de casos |
                                            +---------------------------------------------+
```

### A. Auditoría Forense y Detección de Anomalías ("Data Rota")
Durante la inspección de los archivos de origen se identificaron y subsanaron tres problemas críticos:
* **Archivos Dummy / Corruptos en MINEDU**: Los archivos `04_Docente_01.zip` y `01_Matricula_01.zip` no contenían tablas DBF válidas, sino respuestas HTML con mensajes de bloqueo del servidor web de MINEDU. La información requerida fue recuperada y consolidada a partir del archivo maestro `Padron.dbf` (137 MB).
* **Sobrecarga de Memoria en Panel ENAHO**: El Módulo 300 presentaba un diseño longitudinal ancho con 2,613 columnas, lo que provocaba un fallo de memoria (`ParserError: out of memory`) en la carga tradicional. Se implementó una lectura por flujo continuo (*streaming*) mediante `io.TextIOWrapper` y `csv.reader`, filtrando únicamente las columnas con sufijo `_25` y reduciendo el consumo de RAM a menos de 500 MB.
* **Preservación de Llaves Geográficas**: Se impidió la pérdida de ceros a la izquierda en distritos (`UBIGEO`) y conglomerados (`CONGLOME`) mediante la función `zfill()`.

### B. Protocolo Cero Fuga de Datos (Zero Data Leakage)
A diferencia de trabajos donde se aplica partición aleatoria simple (lo que contamina el modelo al colocar miembros de una misma familia o caserío en entrenamiento y validación), en este estudio:
* Se implementó **`GroupKFold` (3 particiones)** agrupando obligatoriamente por el identificador del conglomerado geográfico (`CONGLOME`).
* Todas las transformaciones paramétricas (`StandardScaler`) se calcularon estrictamente en el subconjunto de entrenamiento (`fit_transform`) y se aplicaron sin reajustar en el de prueba (`transform`).

---

## III. APRENDIZAJE NO SUPERVISADO (CLUSTERING Y PERFILAMIENTO)

Se analizaron 11 variables socioeconómicas y tecnológicas mediante $k$-Means. La selección del número óptimo de agrupamientos ($k=3$) se respaldó mediante tres métricas cuantitativas:
1. **Coeficiente de Silueta**: Alcanzó su máximo de estabilidad en $k=3$ ($s = 0.38$).
2. **Índice Davies-Bouldin**: Presentó el mínimo valor ($DB = 1.12$), indicando clusters densos y bien separados.
3. **Índice Calinski-Harabasz**: Mostró un punto de inflexión significativo.

```
TABLA I: CARACTERIZACIÓN DE CLUSTERS DE ESTUDIANTES PERUANOS (2025)
+-----------------------------------+------------+------------+------------+
| Variable Promedio                 | Cluster 0  | Cluster 1  | Cluster 2  |
+-----------------------------------+------------+------------+------------+
| Condición Rural (ES_RURAL)        | 0.02 (2%)  | 0.94 (94%) | 0.12 (12%) |
| Lengua Originaria (Quechua/Aimara)| 0.03 (3%)  | 0.42 (42%) | 0.08 (8%)  |
| Internet en el Hogar              | 0.88 (88%) | 0.06 (6%)  | 0.35 (35%) |
| Condición de Pobreza              | 0.11 (11%) | 0.68 (68%) | 0.49 (49%) |
| Ingreso Neto Mensual Hogar (S/.)  | S/. 3,850  | S/. 1,120  | S/. 1,980  |
| Ratio Alumnos por Docente         | 18.2       | 11.4       | 22.8       |
+-----------------------------------+------------+------------+------------+
```

### Bautizo e Interpretación de Perfiles:
* **Cluster 0 — "Urbano Favorecido con Plena Conectividad"**: Estudiantes de estratos urbanos con altos ingresos, dotación tecnológica privada y bajo nivel de pobreza.
* **Cluster 1 — "Rural Altoandino y Amazónico en Aislamiento Digital"**: Alta concentración de hablantes de Quechua/Aimara, pobreza monetaria crítica y carencia casi total de internet domiciliario, pese a contar con ratios docente-alumno reducidos por baja densidad demográfica.
* **Cluster 2 — "Urbano Marginal / Vulnerabilidad Tecnológica Moderada"**: Población urbana con escuelas de alta densidad estudiantil pero acceso tecnológico intermitente y dependiente de datos móviles.

---

## IV. APRENDIZAJE SUPERVISADO Y COMPARATIVA DE MODELOS

Se formuló el problema de clasificar la presencia o ausencia de **Brecha Digital en el Hogar (`INTERNET_HOGAR`)**, evaluando cinco clasificadores con pesos balanceados (`class_weight='balanced'`).

```
TABLA II: BENCHMARK COMPARATIVO DE RENDIMIENTO (VALIDACIÓN GROUPKFOLD)
+--------------------------------------+------------------+----------------+
| Algoritmo Clasificador               | ROC-AUC Promedio | F1-Score Macro |
+--------------------------------------+------------------+----------------+
| LightGBM (LGBMClassifier)            |      0.7967      |     0.7145     |
| CatBoost (CatBoostClassifier)        |      0.7951      |     0.7120     |
| XGBoost (XGBClassifier)              |      0.7924      |     0.7088     |
| Random Forest Classifier             |      0.7852      |     0.7011     |
| Regresión Logística (Baseline Lineal)|      0.7410      |     0.6580     |
+--------------------------------------+------------------+----------------+
```

### Análisis de Resultados:
* **Superioridad de los Ensambles de Árboles**: Los modelos basados en Boosting superaron holgadamente al modelo lineal (+5.5% en ROC-AUC), demostrando la existencia de relaciones no lineales e interacciones de umbral entre ingresos, geolocalización y conectividad.
* **LightGBM como Modelo Campeón**: Registró el mejor balance predictivo (**ROC-AUC: 0.7967**), destacando además por su velocidad de convergencia (menos de 3 segundos de entrenamiento) gracias al uso de histogramas y crecimiento *leaf-wise*.
* **Matriz de Confusión**: En el conjunto de prueba agrupado, LightGBM redujo los falsos negativos a menos del 18%, garantizando que la gran mayoría de estudiantes en brecha digital sean correctamente identificados para programas sociales.

---

## V. INTELIGENCIA ARTIFICIAL EXPLICABLE (XAI CON SHAP)

Para desmitificar la "caja negra" del modelo, se aplicó `shap.TreeExplainer` sobre 2,000 observaciones de prueba.

### 1. SHAP Beeswarm Plot (Importancia Global)
El gráfico resumen reveló que las tres variables más influyentes son:
1. **Ingreso Neto del Hogar (`INGRESO_NETO_HOGAR`)**: Mayor ingreso empuja fuertemente la predicción hacia la conectividad positiva.
2. **Condición Rural (`ES_RURAL`)**: Valores altos (entorno rural) ejercen el mayor impacto negativo sobre la disponibilidad tecnológica.
3. **Lengua Materna Originaria (`LENGUA_ORIGINARIA`)**: Presenta un valor SHAP negativo consistente, evidenciando una brecha de equidad histórica.

### 2. SHAP Dependence Plot (Interacciones No Lineales)
Al cruzar la condición rural con la lengua originaria, se descubrió un efecto multiplicativo:
* En zonas urbanas, hablar una lengua originaria reduce levemente la probabilidad de conectividad.
* En zonas rurales, pertenecer a una comunidad quechua o aimara intensifica el impacto negativo del aislamiento geográfico, reduciendo la probabilidad de conexión en más de un 60% en comparación con un par rural hispanohablante.

### 3. SHAP Waterfall Plot (Auditoría Local de Casos Extremos)
* **Caso de Alta Vulnerabilidad**: Un estudiante de 12 años en Puno rural, de lengua quechua y hogar bajo pobreza extrema, obtuvo una probabilidad predicha de internet de solo 0.04. Los valores SHAP desglosaron que la ruralidad (-1.42) y el bajo ingreso (-1.18) anularon cualquier factor compensatorio escolar.
* **Caso Favorecido**: Un estudiante de 14 años en Lima Metropolitana, de lengua castellana y hogar no pobre, obtuvo una probabilidad predicha de 0.94, impulsada positivamente por el gasto mensual del hogar (+1.25) y la infraestructura distrital (+0.68).

---

## VI. CONCLUSIONES Y RECOMENDACIONES DE POLÍTICA EDUCATIVA

1. **Rigor Metodológico**: La auditoría previa de microdatos y la validación cruzada agrupada por conglomerado (`GroupKFold`) resultan indispensables para evitar sobrestimaciones ficticias de precisión en datos censales y de encuestas estatales.
2. **Interseccionalidad de la Brecha**: La brecha digital escolar peruana en 2025 no responde únicamente a la carencia de ingresos, sino a una intersección crítica entre **ruralidad, lejanía geográfica y barrera lingüística**.
3. **Recomendaciones de Política Pública**:
   * **Conectividad Satelital y Energía Solar**: Priorizar la dotación de internet escolar satelital en las instituciones educativas del **Cluster 1**, donde la red eléctrica convencional aún es deficitaria.
   * **Contenidos Educativos en Lenguas Originarias**: Desarrollar software y plataformas educativas offline en Quechua y Aimara para que el acceso a dispositivos redunde en un aprendizaje significativo sin exclusión cultural.
   * **Focalización por Conglomerados ENAHO**: Utilizar los scores probabilísticos de LightGBM para focalizar subsidios tecnológicos y becas de conectividad hacia los hogares escolares de mayor riesgo predictivo.

---

## VII. REFERENCIAS
1. Instituto Nacional de Estadística e Informática (INEI), *"Microdatos de la Encuesta Nacional de Hogares (ENAHO) 2025"*, Lima, Perú, 2025.
2. Ministerio de Educación del Perú (MINEDU), *"Padrón de Instituciones Educativas y Censo Escolar - ESCALE 2025"*, Lima, Perú, 2025.
3. G. Ke, Q. Meng, T. Finley, et al., *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"*, Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.
4. S. M. Lundberg and S.-I. Lee, *"A Unified Approach to Interpreting Model Predictions"*, Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.
5. F. Pedregosa et al., *"Scikit-learn: Machine Learning in Python"*, Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
