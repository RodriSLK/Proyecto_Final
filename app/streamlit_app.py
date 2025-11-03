# app/streamlit_app.py

from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

TARGET_COL = "clase_general"


# ================================
# Carga de modelo y metadatos
# ================================
@st.cache_resource
def load_model():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = models_dir / "best_model.pkl"
    model = joblib.load(model_path)
    return model


@st.cache_resource
def load_metadata():
    models_dir = Path(__file__).resolve().parent.parent / "models"

    skills_path = models_dir / "skills_cols.json"
    labels_path = models_dir / "class_labels.json"

    with open(skills_path, "r", encoding="utf-8") as f:
        skills_cols = json.load(f)

    with open(labels_path, "r", encoding="utf-8") as f:
        class_labels = json.load(f)

    return skills_cols, class_labels


@st.cache_data
def load_data():
    """
    Carga el CSV limpio con las ofertas.
    Ajustá el nombre si tu archivo se llama distinto.
    """
    data_path = Path(__file__).resolve().parent.parent / "computrabajo_2025-10-17_limpio_full.csv"
    df = pd.read_csv(data_path)
    return df


@st.cache_data
def load_results():
    """
    Carga el CSV con los resultados de test:
    columnas 'real' y 'predicho'.
    """
    results_path = Path(__file__).resolve().parent.parent / "resultados_test.csv"
    df_res = pd.read_csv(results_path)
    return df_res


# ================================
# Helpers para frecuencias de skills
# ================================
def compute_skill_freq(df, skill_cols, target_col, selected_class=None, top_n=15):
    """
    Devuelve un DataFrame con columnas:
    - skill
    - freq  (proporción de ofertas donde esa skill = 1)
    filtrado por clase si corresponde.
    """
    if selected_class and selected_class != "Todos":
        df_sub = df[df[target_col] == selected_class].copy()
    else:
        df_sub = df.copy()

    if df_sub.empty:
        return pd.DataFrame(columns=["skill", "freq"])

    freq = df_sub[skill_cols].mean().sort_values(ascending=False)
    freq = freq.reset_index()
    freq.columns = ["skill", "freq"]

    return freq.head(top_n)


# ================================
# Páginas / Secciones
# ================================
def page_home(class_labels, skills_cols):
    st.title("💼 Proyecto Final - Clasificación de perfiles IT")
    st.markdown(
        """
        Esta app forma parte de la **Entrega 4 – Visualización e Integración**.

        - Usa un modelo entrenado de **Gradient Boosting con GridSearch** para predecir la `clase_general` de una oferta IT.
        - Se basa en la presencia/ausencia de **hard skills** y **soft skills** codificadas como variables binarias.
        - La idea es explorar los datos, ver el rendimiento del modelo y permitir probar predicciones con nuevas combinaciones de skills.
        """
    )

    st.subheader("Clases que el modelo puede predecir")
    st.write(class_labels)

    st.subheader("Cantidad de columnas de skills")
    st.write(f"{len(skills_cols)} columnas")

    with st.expander("Ver lista completa de skills"):
        st.write(skills_cols)


def page_exploracion(df, skills_cols, class_labels):
    st.title("📊 Exploración de datos")

    st.markdown(
        """
        En esta sección exploramos cómo se distribuyen las **skills** en las ofertas,
        diferenciando entre **soft skills** y **hard skills** según el tipo de puesto (`clase_general`).
        """
    )

    soft_cols = [c for c in skills_cols if c.startswith("soft_")]
    hard_cols = [c for c in skills_cols if not c.startswith("soft_")]

    clases = sorted(df[TARGET_COL].dropna().unique())
    selected_class = st.selectbox(
        "Elegí el tipo de puesto para analizar (o 'Todos'):",
        options=["Todos"] + list(clases),
    )

    st.markdown(
        f"Mostrando proporción de ofertas con cada skill para: **{selected_class}**."
        if selected_class != "Todos"
        else "Mostrando proporción de ofertas con cada skill para **todas las clases**."
    )

    tab_soft, tab_hard = st.tabs(["🧠 Soft skills", "💻 Hard skills"])

    # --------- SOFT SKILLS ----------
    with tab_soft:
        st.subheader("Distribución de soft skills")
        if not soft_cols:
            st.warning("No se encontraron columnas de soft skills (prefijo 'soft_').")
        else:
            soft_freq = compute_skill_freq(df, soft_cols, TARGET_COL, selected_class, top_n=15)

            if soft_freq.empty:
                st.warning("No hay datos para esa combinación de filtros.")
            else:
                chart_soft = (
                    alt.Chart(soft_freq)
                    .mark_bar()
                    .encode(
                        x=alt.X("skill:N", sort="-y", title="Soft skill"),
                        y=alt.Y("freq:Q", title="Proporción de ofertas"),
                        tooltip=[
                            alt.Tooltip("skill:N", title="Soft skill"),
                            alt.Tooltip("freq:Q", title="Proporción", format=".2f"),
                        ],
                    )
                    .properties(height=350)
                )

                st.altair_chart(chart_soft, use_container_width=True)

    # --------- HARD SKILLS ----------
    with tab_hard:
        st.subheader("Distribución de hard skills")
        if not hard_cols:
            st.warning("No se encontraron columnas de hard skills.")
        else:
            hard_freq = compute_skill_freq(df, hard_cols, TARGET_COL, selected_class, top_n=15)

            if hard_freq.empty:
                st.warning("No hay datos para esa combinación de filtros.")
            else:
                chart_hard = (
                    alt.Chart(hard_freq)
                    .mark_bar()
                    .encode(
                        x=alt.X("skill:N", sort="-y", title="Hard skill"),
                        y=alt.Y("freq:Q", title="Proporción de ofertas"),
                        tooltip=[
                            alt.Tooltip("skill:N", title="Hard skill"),
                            alt.Tooltip("freq:Q", title="Proporción", format=".2f"),
                        ],
                    )
                    .properties(height=350)
                )

                st.altair_chart(chart_hard, use_container_width=True)


def page_rendimiento(df_res, class_labels):
    st.title("📉 Rendimiento del modelo")

    st.markdown(
        """
        A continuación se muestran las métricas de rendimiento del modelo
        sobre el conjunto de **test**, junto con la **matriz de confusión**.
        """
    )

    if df_res.empty:
        st.warning("El archivo de resultados de test está vacío.")
        return

    y_true = df_res["real"]
    y_pred = df_res["predicho"]

    # Métricas globales
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy (test)", f"{acc:.3f}")
    with col2:
        st.metric("F1-macro (test)", f"{f1:.3f}")

    st.markdown("### Matriz de confusión")

    # Matriz de confusión con orden consistente de clases
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    # Pasar a formato largo (long) para Altair
    cm_long = (
        cm_df.reset_index()
        .melt(id_vars="index", var_name="Predicha", value_name="count")
        .rename(columns={"index": "Real"})
    )

    # Heatmap de fondo (colores)
    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicha:N", title="Clase predicha"),
            y=alt.Y("Real:N", title="Clase real"),
            color=alt.Color("count:Q", title="Cantidad"),
            tooltip=[
                alt.Tooltip("Real:N", title="Clase real"),
                alt.Tooltip("Predicha:N", title="Clase predicha"),
                alt.Tooltip("count:Q", title="Cantidad"),
            ],
        )
        .properties(height=400)
    )

    # Números encima de cada celda
    text = (
        alt.Chart(cm_long)
        .mark_text(baseline="middle")
        .encode(
            x="Predicha:N",
            y="Real:N",
            text=alt.Text("count:Q", format=".0f"),
        )
    )

    chart_cm = heatmap + text

    st.altair_chart(chart_cm, use_container_width=True)

    st.caption(
        "La matriz de confusión muestra en el eje vertical las clases reales y en el eje horizontal "
        "las clases predichas. Cada celda indica cuántos casos caen en esa combinación."
    )


def build_input_from_skills(selected_skills, skills_cols):
    """
    Crea un DataFrame de una fila con 0/1 según las skills seleccionadas.
    - selected_skills: lista de nombres de columnas (skills) activas.
    - skills_cols: lista completa de columnas que espera el modelo.
    """
    row = {col: 0 for col in skills_cols}
    for sk in selected_skills:
        if sk in row:
            row[sk] = 1
    return pd.DataFrame([row])



def page_prediccion(model, class_labels, skills_cols):
    st.title("🧠 Predicción y comportamiento del modelo")
    st.markdown(
        """
        Construí un perfil seleccionando **hard skills** y **soft skills** y luego hacé clic en **Predecir**.

        La app arma internamente un vector de 0/1 con las skills elegidas y lo pasa al modelo
        para obtener la `clase_general` más probable y las probabilidades asociadas a cada clase.
        """
    )

    # Separar soft y hard skills
    soft_cols = [c for c in skills_cols if c.startswith("soft_")]
    hard_cols = [c for c in skills_cols if not c.startswith("soft_")]

    col_soft, col_hard = st.columns(2)

    with col_soft:
        with st.expander("🧠 Soft skills (click para desplegar)", expanded=False):
            selected_soft = st.multiselect(
                "Elegí soft skills para el perfil:",
                options=soft_cols,
                format_func=lambda s: s.replace("soft_", "").replace("_", " ").capitalize(),
            )

    with col_hard:
        with st.expander("💻 Hard skills (click para desplegar)", expanded=False):
            selected_hard = st.multiselect(
                "Elegí hard skills para el perfil:",
                options=hard_cols,
            )

    selected_skills = selected_soft + selected_hard

    st.markdown("### Perfil armado")
    if selected_skills:
        col_ps, col_ph = st.columns(2)
        with col_ps:
            st.write("🧠 Soft skills seleccionadas:")
            if selected_soft:
                st.write(
                    [
                        s.replace("soft_", "").replace("_", " ").capitalize()
                        for s in selected_soft
                    ]
                )
            else:
                st.write("Ninguna")

        with col_ph:
            st.write("💻 Hard skills seleccionadas:")
            st.write(selected_hard or "Ninguna")
    else:
        st.info("Todavía no seleccionaste ninguna skill. Usá los paneles de arriba para armar el perfil.")

    st.markdown("### Predicción del modelo")

    if st.button("🔮 Predecir perfil de puesto"):
        if not selected_skills:
            st.warning("Por favor, seleccioná al menos una skill antes de predecir.")
            return

        # Armar vector de entrada 0/1
        X_new = build_input_from_skills(selected_skills, skills_cols)

        # Predicción de clase y probabilidades
        try:
            pred_class = model.predict(X_new)[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_new)[0]
                df_probs = pd.DataFrame({
                    "clase": class_labels,
                    "prob": probs
                })
            else:
                probs = None
                df_probs = None
        except Exception as e:
            st.error("❌ Ocurrió un error al realizar la predicción.")
            st.exception(e)
            return

        st.subheader("Resultado de la predicción")
        st.success(f"Clase general predicha: **{pred_class}**")

        if probs is not None and df_probs is not None:
            st.markdown("### Probabilidades por clase")

            chart_probs = (
                alt.Chart(df_probs)
                .mark_bar()
                .encode(
                    x=alt.X("clase:N", title="Clase"),
                    y=alt.Y("prob:Q", title="Probabilidad"),
                    tooltip=[
                        alt.Tooltip("clase:N", title="Clase"),
                        alt.Tooltip("prob:Q", title="Probabilidad", format=".2f"),
                    ],
                )
                .properties(height=300)
            )

            st.altair_chart(chart_probs, use_container_width=True)

            st.caption(
                "La clase mostrada arriba es la de mayor probabilidad. "
                "Las barras muestran cómo reparte el modelo la probabilidad entre las clases posibles."
            )

    with st.expander("Información técnica del modelo"):
        st.write("Tipo de modelo:", type(model).__name__)
        st.write("Cantidad de clases:", len(class_labels))

# ================================
# App principal
# ================================
def main():
    st.set_page_config(
        page_title="Proyecto IT - Clasificación de ofertas",
        page_icon="💼",
        layout="wide",
    )

    try:
        model = load_model()
        skills_cols, class_labels = load_metadata()
        df = load_data()
        df_res = load_results()
    except Exception as e:
        st.error("❌ Hubo un problema al cargar el modelo, metadatos o datos.")
        st.exception(e)
        return

    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Ir a:",
        (
            "🏠 Inicio",
            "📊 Exploración de datos",
            "📉 Rendimiento del modelo",
            "🧠 Predicción y comportamiento",
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Entrega 4 – Visualización e Integración")

    if page == "🏠 Inicio":
        page_home(class_labels, skills_cols)
    elif page == "📊 Exploración de datos":
        page_exploracion(df, skills_cols, class_labels)
    elif page == "📉 Rendimiento del modelo":
        page_rendimiento(df_res, class_labels)
    elif page == "🧠 Predicción y comportamiento":
        page_prediccion(model, class_labels, skills_cols)


if __name__ == "__main__":
    main()
