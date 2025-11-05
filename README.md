# 💼 Proyecto Final – Clasificación de Perfiles IT

Este proyecto forma parte de la **Entrega 4 – Visualización e Integración**, correspondiente a la materia de Ciencia de Datos.  
El objetivo es construir una aplicación interactiva que integre todo el trabajo realizado en las etapas anteriores (análisis, modelado y evaluación) y permita **explorar, visualizar y probar el modelo predictivo** entrenado.

---

## 🚀 Descripción del proyecto

El modelo predice la variable **`clase_general`** (tipo de puesto IT: Analista, Desarrollador, QA, etc.)  
a partir de la presencia/ausencia de **hard skills** y **soft skills** en las ofertas laborales.

Para ello se utilizó un pipeline completo en *Scikit-learn* con un **Gradient Boosting optimizado mediante GridSearchCV**, alcanzando un buen equilibrio entre rendimiento y generalización.

La app final fue desarrollada con **Streamlit** y **Altair**, integrando:

- Exploración interactiva de datos (frecuencia de skills por clase).  
- Evaluación del modelo (métricas y matriz de confusión).  
- Interfaz para probar nuevas combinaciones de skills y obtener predicciones.

---

## 🧩 Estructura del proyecto

```text
Proyecto_Final/
├── app/
│   └── streamlit_app.py              # App principal de Streamlit
├── models/
│   ├── best_model.pkl                # Modelo final entrenado (Gradient Boosting)
│   ├── skills_cols.json              # Lista de columnas usadas por el modelo
│   └── class_labels.json             # Nombres de las clases objetivo
├── resultados_test.csv               # Resultados reales vs predichos (test)
├── computrabajo_2025-10-17_limpio_full.csv   # Dataset limpio
├── Entrega3.ipynb                    # Notebook de modelado y evaluación
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo

```
---

## ⚙️ Cómo ejecutar la aplicación localmente

1. Cloná el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/Proyecto_Final.git
   cd Proyecto_Final

2. Creá y activá un entorno virtual (opcional pero recomendado):

     python -m venv .venv
    .venv\Scripts\activate    # En Windows
    source .venv/bin/activate # En Linux/Mac

3. Instalá las dependencias:

    pip install -r requirements.txt

4. Ejecutá la aplicación:

    streamlit run app/streamlit_app.py

5. Abrí el enlace local que aparece en la consola (por defecto http://localhost:8501).

---

## 🖼️ Secciones de la aplicación

### 🏠 **Inicio**
Presenta la descripción general del proyecto, el tipo de modelo utilizado y las clases que puede predecir.


### 📊 **Exploración de datos**
Incluye visualizaciones interactivas construidas con **Altair**:
- 📈 Frecuencia de *soft skills* por tipo de puesto.  
- 💻 Frecuencia de *hard skills* por tipo de puesto.

Permite comparar qué habilidades predominan según la variable `clase_general`.


### 📉 **Rendimiento del modelo**
Muestra:
- **Métricas globales:** *Accuracy* y *F1-macro*.  
- **Matriz de confusión interactiva:** con colores y conteos por celda, para identificar las clases que el modelo predice mejor o confunde más.


### 🧠 **Predicción y comportamiento**
Ofrece una interfaz sencilla para construir un perfil seleccionando *hard* y *soft skills*.  
El modelo predice la **clase general** correspondiente y muestra un **gráfico de barras** con las probabilidades por clase.
