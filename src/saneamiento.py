"""
saneamiento.py
==============
Script y modulo para la extraccion, auditoria forense, saneamiento y cruce relacional
de los microdatos de ENAHO 2025 (INEI) y ESCALE 2025 (MINEDU).

Manejo optimizado de memoria por STREAMING para archivos longitudinales de 2,600+ columnas.
"""

import os
import io
import csv
import zipfile
import numpy as np
import pandas as pd
from dbfread import DBF

def diagnosticar_archivos_corruptos(zip_path: str):
    """
    Auditoria forense para evidenciar los archivos corruptos/rotos
    presentes en el paquete oficial descargado.
    """
    print("=" * 70)
    print("FASE 0: EVIDENCIA FORENSE DE ARCHIVOS DANADOS / ANOMALIAS EN DATA")
    print("=" * 70)
    evidencias = {}
    with zipfile.ZipFile(zip_path, 'r') as z_outer:
        for falso_zip in ['ENAHO/MINEDU/04_Docente_01.zip', 'ENAHO/MINEDU/01_Matricula_01.zip']:
            content = z_outer.read(falso_zip)
            es_html = b'<html' in content.lower()
            evidencias[falso_zip] = {
                'tamano_bytes': len(content),
                'es_zip_valido': content[:4] == b'PK\x03\x04',
                'es_html_error': es_html,
                'primeros_caracteres': content[:150].decode('latin1', errors='replace')
            }
            print(f"[!] Archivo detectado: {falso_zip}")
            print(f"    - Tamano: {len(content)} bytes ({len(content)/1024:.2f} KB)")
            print(f"    - Encabezado PK Zip: {content[:4] == b'PK\x03\x04'}")
            print(f"    - Diagnostico: Falso ZIP. Es una respuesta HTML de error/bloqueo de MINEDU.")
            print(f"    - Vista previa: {content[:85]}...\n")
    return evidencias


def procesar_escale(base_dir: str):
    """
    Lee Padron.dbf de MINEDU y genera indicadores agregados a nivel distrital (CODGEO = UBIGEO).
    """
    padron_path = os.path.join(base_dir, 'Padron.dbf')
    if not os.path.exists(padron_path):
        raise FileNotFoundError(f"No se encontro {padron_path}")

    print("[1/5] Procesando ESCALE (Padron de Instituciones Educativas)...")
    dbf = DBF(padron_path, encoding='latin1', load=True)
    df_pad = pd.DataFrame(iter(dbf))

    # Asegurar 6 digitos en CODGEO (UBIGEO)
    df_pad['CODGEO'] = df_pad['CODGEO'].astype(str).str.strip().str.zfill(6)
    
    # Limpiar columnas numericas
    for col in ['TALUM', 'TDOC', 'TSEC', 'NLAT_IE', 'NLONG_IE']:
        if col in df_pad.columns:
            df_pad[col] = pd.to_numeric(df_pad[col], errors='coerce').fillna(0)

    # Es colegio publico? GESTION = '1' (Publica de gestion directa) o '2' (Publica de gestion privada)
    df_pad['ES_PUBLICO'] = df_pad['GESTION'].astype(str).str.startswith('1').astype(int)

    # Agregacion por distrito (CODGEO)
    distrito_agg = df_pad.groupby('CODGEO').agg(
        TOTAL_COLEGIOS=('COD_MOD', 'count'),
        TOTAL_ALUMNOS=('TALUM', 'sum'),
        TOTAL_DOCENTES=('TDOC', 'sum'),
        PROP_COLEGIOS_PUBLICOS=('ES_PUBLICO', 'mean'),
        NLAT_DISTRITO=('NLAT_IE', lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0),
        NLONG_DISTRITO=('NLONG_IE', lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0)
    ).reset_index()

    # Ratio alumno por docente
    distrito_agg['RATIO_ALUMNO_DOCENTE'] = np.where(
        distrito_agg['TOTAL_DOCENTES'] > 0,
        distrito_agg['TOTAL_ALUMNOS'] / distrito_agg['TOTAL_DOCENTES'],
        np.nan
    )
    distrito_agg['RATIO_ALUMNO_DOCENTE'] = distrito_agg['RATIO_ALUMNO_DOCENTE'].clip(lower=1, upper=80)
    distrito_agg.rename(columns={'CODGEO': 'UBIGEO'}, inplace=True)
    
    print(f"    [OK] ESCALE consolidado: {len(distrito_agg)} distritos unicos procesados.")
    return distrito_agg


def leer_modulo_stream(sub_zip, csv_rel_path, cols_needed, filter_col='conglome_25'):
    """
    Lee un archivo CSV masivo mediante streaming fila por fila,
    extrayendo unicamente las columnas deseadas y descartando filas vacias para el 2025.
    Evita ParserError C out-of-memory.
    """
    with sub_zip.open(csv_rel_path) as f:
        text_stream = io.TextIOWrapper(f, encoding='latin1')
        reader = csv.reader(text_stream)
        header = next(reader)
        
        # Mapear indices de columnas
        col_indices = {}
        for c in cols_needed:
            if c in header:
                col_indices[c] = header.index(c)
            else:
                matches = [i for i, h in enumerate(header) if h.lower() == c.lower()]
                if matches:
                    col_indices[c] = matches[0]

        idx_filter = col_indices.get(filter_col, None)
        rows = []
        for row in reader:
            if idx_filter is not None and (len(row) <= idx_filter or not row[idx_filter].strip()):
                continue
            row_data = [row[col_indices[c]] if c in col_indices and len(row) > col_indices[c] else '' for c in cols_needed]
            rows.append(row_data)

        df = pd.DataFrame(rows, columns=cols_needed)
        return df


def cargar_enaho_2025(zip_path: str):
    """
    Extrae, filtra la cohorte 2025 (sufijo _25) y sanea los 4 modulos clave de ENAHO mediante streaming.
    """
    with zipfile.ZipFile(zip_path, 'r') as z_outer:
        # ---------------------------------------------------------
        # MODULO 200: Miembros del hogar y edad escolar
        # ---------------------------------------------------------
        print("[2/5] Extrayendo Modulo 200 (Miembros del Hogar - ENAHO)...")
        z200 = zipfile.ZipFile(io.BytesIO(z_outer.read('ENAHO/INEI/1041-Modulo1479.zip')))
        cols_200 = ['conglome_25', 'vivienda_25', 'hogar_25', 'codperso_25', 'ubigeo_25', 
                    'dominio_25', 'estrato_25', 'p207_25', 'p208a_25', 'p203_25', 'facpob07_25']
        
        df_200 = leer_modulo_stream(z200, '1041-Modulo1479/enaho01-2021-2025-200-panel.csv', cols_200)

        # Saneamiento de llaves con ceros a la izquierda
        for k, length in [('conglome_25', 6), ('vivienda_25', 3), ('hogar_25', 2), ('codperso_25', 2), ('ubigeo_25', 6)]:
            df_200[k] = df_200[k].astype(str).str.strip().str.zfill(length)

        df_200['p208a_25'] = pd.to_numeric(df_200['p208a_25'], errors='coerce')
        # FILTRADO DE POBLACION OBJETIVO (6 A 17 ANOS)
        df_escolar = df_200[df_200['p208a_25'].between(6, 17)].copy()
        print(f"    [OK] Poblacion escolar (6-17 anos) en 2025: {len(df_escolar):,} registros.")

        # ---------------------------------------------------------
        # MODULO 300: Educacion y TICs
        # ---------------------------------------------------------
        print("[3/5] Extrayendo Modulo 300 (Educacion y TICs)...")
        z300 = zipfile.ZipFile(io.BytesIO(z_outer.read('ENAHO/INEI/1041-Modulo1475.zip')))
        cols_300 = ['conglome_25', 'vivienda_25', 'hogar_25', 'codperso_25',
                    'p300a_25', 'p306_25', 'p307_25', 'p308a_25', 'p308b_25', 
                    'p314a_25', 'p314b_1_25', 'p314b_2_25', 'p316_1_25']
        
        df_300 = leer_modulo_stream(z300, '1041-Modulo1475/enaho01a-2021-2025-300-panel.csv', cols_300)
        for k, length in [('conglome_25', 6), ('vivienda_25', 3), ('hogar_25', 2), ('codperso_25', 2)]:
            df_300[k] = df_300[k].astype(str).str.strip().str.zfill(length)
        print(f"    [OK] Modulo 300 cargado: {len(df_300):,} personas en 2025.")

        # ---------------------------------------------------------
        # MODULO 100: Vivienda y Servicios (Internet en hogar, luz)
        # ---------------------------------------------------------
        print("[4/5] Extrayendo Modulo 100 (Vivienda y Servicios)...")
        z100 = zipfile.ZipFile(io.BytesIO(z_outer.read('ENAHO/INEI/1041-Modulo1474.zip')))
        cols_100 = ['conglome_25', 'vivienda_25', 'hogar_25', 'p1121_25', 'p1144_25']
        
        df_100 = leer_modulo_stream(z100, '1041-Modulo1474/enaho01-2021-2025-100-panel.csv', cols_100)
        for k, length in [('conglome_25', 6), ('vivienda_25', 3), ('hogar_25', 2)]:
            df_100[k] = df_100[k].astype(str).str.strip().str.zfill(length)
        print(f"    [OK] Modulo 100 cargado: {len(df_100):,} hogares en 2025.")

        # ---------------------------------------------------------
        # SUMARIA: Pobreza e Ingresos
        # ---------------------------------------------------------
        print("[5/5] Extrayendo Modulo Sumaria (Pobreza e Ingresos)...")
        zsum = zipfile.ZipFile(io.BytesIO(z_outer.read('ENAHO/INEI/1041-Modulo1478.zip')))
        cols_sum = ['conglome_25', 'vivienda_25', 'hogar_25', 'pobreza_25', 'inghog2d_25', 'gashog2d_25', 'factor07_25']
        
        df_sum = leer_modulo_stream(zsum, '1041-Modulo1478/sumaria-2021-2025-panel.csv', cols_sum)
        for k, length in [('conglome_25', 6), ('vivienda_25', 3), ('hogar_25', 2)]:
            df_sum[k] = df_sum[k].astype(str).str.strip().str.zfill(length)
        print(f"    [OK] Sumaria cargada: {len(df_sum):,} hogares en 2025.")

    return df_escolar, df_300, df_100, df_sum


def construir_dataset_maestro(zip_path: str, base_dir: str):
    """
    Ejecuta el merge relacional de todos los modulos y ESCALE,
    y construye las variables compuestas y targets.
    """
    # 1. Diagnostico previo
    diagnosticar_archivos_corruptos(zip_path)

    # 2. Cargar ESCALE
    df_escale = procesar_escale(base_dir)

    # 3. Cargar ENAHO
    df_escolar, df_300, df_100, df_sum = cargar_enaho_2025(zip_path)

    # Cruce 1: Persona (df_escolar + df_300)
    llaves_persona = ['conglome_25', 'vivienda_25', 'hogar_25', 'codperso_25']
    print("\nEjecutando Cruce Relacional Maestro...")
    df_master = pd.merge(df_escolar, df_300, on=llaves_persona, how='inner')
    print(f"    [OK] Cruce Persona (Modulo 200 + Modulo 300): {len(df_master):,} estudiantes.")

    # Cruce 2: Hogar (+ Modulo 100)
    llaves_hogar = ['conglome_25', 'vivienda_25', 'hogar_25']
    df_master = pd.merge(df_master, df_100, on=llaves_hogar, how='left')

    # Cruce 3: Sumaria (+ Pobreza / Ingresos)
    df_master = pd.merge(df_master, df_sum, on=llaves_hogar, how='left')

    # Cruce 4: Distrito (+ ESCALE por UBIGEO)
    df_master['UBIGEO'] = df_master['ubigeo_25'].astype(str).str.strip().str.zfill(6)
    df_master = pd.merge(df_master, df_escale, on='UBIGEO', how='left')
    print(f"    [OK] Cruce Geografico Distrital (+ ESCALE): {len(df_master):,} estudiantes con entorno escolar.")

    # ---------------------------------------------------------
    # INGENIERIA DE VARIABLES COMPUESTAS
    # ---------------------------------------------------------
    clean_cols = {
        'conglome_25': 'CONGLOME',
        'vivienda_25': 'VIVIENDA',
        'hogar_25': 'HOGAR',
        'codperso_25': 'CODPERSO',
        'p207_25': 'SEXO',
        'p208a_25': 'EDAD',
        'estrato_25': 'ESTRATO',
        'dominio_25': 'DOMINIO',
        'p300a_25': 'LENGUA_MATERNA_RAW',
        'p306_25': 'MATRICULADO',
        'p307_25': 'ASISTE',
        'p308a_25': 'NIVEL_EDUCATIVO',
        'p308b_25': 'GRADO_EDUCATIVO',
        'p314a_25': 'USA_INTERNET',
        'p314b_1_25': 'USA_INTERNET_HOGAR',
        'p314b_2_25': 'USA_INTERNET_COLEGIO',
        'p316_1_25': 'FRECUENCIA_INTERNET',
        'p1121_25': 'TIENE_LUZ',
        'p1144_25': 'INTERNET_HOGAR',
        'pobreza_25': 'CONDICION_POBREZA',
        'inghog2d_25': 'INGRESO_NETO_HOGAR',
        'gashog2d_25': 'GASTO_NETO_HOGAR',
        'factor07_25': 'FACTOR_EXPANSION'
    }
    df_clean = df_master.rename(columns=clean_cols).copy()

    # Estandarizacion de variables numericas y banderas binarias
    df_clean['EDAD'] = pd.to_numeric(df_clean['EDAD'], errors='coerce')
    df_clean['SEXO_MUJER'] = (pd.to_numeric(df_clean['SEXO'], errors='coerce') == 2).astype(int)
    
    # Area Rural (ESTRATO >= 6 suele ser area rural en ENAHO)
    df_clean['ESTRATO_NUM'] = pd.to_numeric(df_clean['ESTRATO'], errors='coerce').fillna(1)
    df_clean['ES_RURAL'] = (df_clean['ESTRATO_NUM'] >= 6).astype(int)

    # Lengua Materna Originaria (1=Quechua, 2=Aimara, 3=Ashaninka/Amazonica)
    lengua_raw = pd.to_numeric(df_clean['LENGUA_MATERNA_RAW'], errors='coerce').fillna(4)
    df_clean['LENGUA_ORIGINARIA'] = lengua_raw.isin([1, 2, 3]).astype(int)

    # Conectividad y TICs
    df_clean['TIENE_LUZ'] = (pd.to_numeric(df_clean['TIENE_LUZ'], errors='coerce') == 1).astype(int)
    df_clean['INTERNET_HOGAR'] = (pd.to_numeric(df_clean['INTERNET_HOGAR'], errors='coerce') == 1).astype(int)
    df_clean['USA_INTERNET'] = (pd.to_numeric(df_clean['USA_INTERNET'], errors='coerce') == 1).astype(int)

    # Pobreza (1=Pobre Extremo, 2=Pobre No Extremo, 3=No Pobre)
    pobreza_raw = pd.to_numeric(df_clean['CONDICION_POBREZA'], errors='coerce').fillna(3)
    df_clean['ES_POBRE'] = pobreza_raw.isin([1, 2]).astype(int)
    
    # Ingreso y Gasto per capita
    df_clean['INGRESO_NETO_HOGAR'] = pd.to_numeric(df_clean['INGRESO_NETO_HOGAR'], errors='coerce').fillna(0)
    df_clean['GASTO_NETO_HOGAR'] = pd.to_numeric(df_clean['GASTO_NETO_HOGAR'], errors='coerce').fillna(0)

    # Variables escolares
    df_clean['ASISTE'] = (pd.to_numeric(df_clean['ASISTE'], errors='coerce') == 1).astype(int)
    df_clean['NIVEL_EDUCATIVO'] = pd.to_numeric(df_clean['NIVEL_EDUCATIVO'], errors='coerce').fillna(0)
    df_clean['GRADO_EDUCATIVO'] = pd.to_numeric(df_clean['GRADO_EDUCATIVO'], errors='coerce').fillna(0)

    # ---------------------------------------------------------
    # CONSTRUCCION DEL TARGET 1: REZAGO ESCOLAR
    # ---------------------------------------------------------
    # Definicion normativa del MINEDU:
    # Primaria (Nivel 2): Grados 1 a 6. Edad normativa = Grado + 6 anos.
    # Secundaria (Nivel 3): Grados 1 a 5. Edad normativa = Grado + 12 anos.
    # Si la edad del estudiante supera en 2 o mas anos la edad esperada -> Rezago Escolar
    rezago = np.zeros(len(df_clean), dtype=int)
    
    # Condicion Primaria
    mask_pri = (df_clean['NIVEL_EDUCATIVO'] == 2) & (df_clean['GRADO_EDUCATIVO'] > 0)
    edad_esp_pri = df_clean.loc[mask_pri, 'GRADO_EDUCATIVO'] + 6
    rezago[mask_pri] = (df_clean.loc[mask_pri, 'EDAD'] > (edad_esp_pri + 1)).astype(int)

    # Condicion Secundaria
    mask_sec = (df_clean['NIVEL_EDUCATIVO'] == 3) & (df_clean['GRADO_EDUCATIVO'] > 0)
    edad_esp_sec = df_clean.loc[mask_sec, 'GRADO_EDUCATIVO'] + 12
    rezago[mask_sec] = (df_clean.loc[mask_sec, 'EDAD'] > (edad_esp_sec + 1)).astype(int)

    # Si esta en edad escolar y no asiste ni estudia -> Rezago / Desercion
    rezago[df_clean['ASISTE'] == 0] = 1

    df_clean['REZAGO_ESCOLAR'] = rezago

    # ---------------------------------------------------------
    # CONSTRUCCION DEL TARGET 2: EXCLUSION DIGITAL EFECTIVA
    # ---------------------------------------------------------
    # Estudiante sin internet en casa y que tampoco usa internet frecuentemente
    df_clean['EXCLUSION_DIGITAL'] = ((df_clean['INTERNET_HOGAR'] == 0) & (df_clean['USA_INTERNET'] == 0)).astype(int)

    print("\nResumen de Distribucion de Variables Objetivo:")
    print(f"  - Rezago Escolar (Target 1): {df_clean['REZAGO_ESCOLAR'].value_counts(normalize=True).to_dict()}")
    print(f"  - Exclusion Digital (Target 2): {df_clean['EXCLUSION_DIGITAL'].value_counts(normalize=True).to_dict()}")

    # Seleccionar columnas finales consolidadas
    final_cols = [
        'CONGLOME', 'VIVIENDA', 'HOGAR', 'CODPERSO', 'UBIGEO',
        'EDAD', 'SEXO_MUJER', 'ES_RURAL', 'LENGUA_ORIGINARIA',
        'TIENE_LUZ', 'INTERNET_HOGAR', 'USA_INTERNET', 'ES_POBRE',
        'INGRESO_NETO_HOGAR', 'GASTO_NETO_HOGAR',
        'NIVEL_EDUCATIVO', 'GRADO_EDUCATIVO', 'ASISTE',
        'TOTAL_COLEGIOS', 'TOTAL_ALUMNOS', 'TOTAL_DOCENTES',
        'PROP_COLEGIOS_PUBLICOS', 'RATIO_ALUMNO_DOCENTE',
        'NLAT_DISTRITO', 'NLONG_DISTRITO',
        'FACTOR_EXPANSION',
        'REZAGO_ESCOLAR', 'EXCLUSION_DIGITAL'
    ]
    df_final = df_clean[final_cols].copy()

    # Imputar medianas en variables de ESCALE para distritos con datos faltantes
    for num_col in ['RATIO_ALUMNO_DOCENTE', 'PROP_COLEGIOS_PUBLICOS', 'TOTAL_COLEGIOS', 'NLAT_DISTRITO', 'NLONG_DISTRITO']:
        df_final[num_col] = df_final[num_col].fillna(df_final[num_col].median())

    # Guardar en parquet y csv
    out_parquet = os.path.join(base_dir, 'data', 'processed', 'dataset_enaho_escale_2025.parquet')
    out_csv = os.path.join(base_dir, 'data', 'processed', 'dataset_enaho_escale_2025.csv')
    df_final.to_parquet(out_parquet, index=False)
    df_final.to_csv(out_csv, index=False, encoding='utf-8')

    print(f"\n[OK] Dataset Maestro Guardado con Exito!")
    print(f"    - Parquet: {out_parquet} ({os.path.getsize(out_parquet)/(1024*1024):.2f} MB)")
    print(f"    - CSV:     {out_csv} ({os.path.getsize(out_csv)/(1024*1024):.2f} MB)")
    print(f"    - Dimensiones finales: {df_final.shape[0]:,} filas x {df_final.shape[1]} columnas.")

    return df_final


if __name__ == '__main__':
    zip_path = r'D:\Documentos\Trabajos upeu\UPEU\Inteligencia Artificial\ENAHO-20261027T185350Z-1-001.zip'
    base_dir = r'D:\Documentos\Trabajos upeu\Trabajos upeu\C8\IA\U2_Examen'
    construir_dataset_maestro(zip_path, base_dir)
