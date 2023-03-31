import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Ajouter les modules nécessaires pour la création du fichier HTML
from io import StringIO
import base64



# Lier le fichier CSS externe, mais ça marche pas 
#st.markdown(<link rel='stylesheet' href='style.css'>, unsafe_allow_html=True)

# Insérer une image
#st.markdown('<img src="data.jpg">', unsafe_allow_html=True)
# Titre de l'application
st.title("Analyse de données Excel")

# Chargement du fichier Excel
file = st.file_uploader("Importer un fichier Excel", type=["xls", "xlsx"])



# Options d'analyse
analyse_options = {
    "Afficher les premières lignes": False,
    "Afficher le nombre de lignes et de colonnes": False,
    "Afficher les statistiques descriptives": False,
    "Analyser la matrice de corrélation": False,
    "Analyser les variables catégorielles": False,
    "Analyser les variables numériques": False,
    "Analyser les données géospatiales": False,

}

if file:
    # Affichage des options d'analyse
    st.sidebar.write("Options d'analyse :")
    for key in analyse_options.keys():
        analyse_options[key] = st.sidebar.checkbox(key)

    # Chargement du fichier dans un DataFrame
    df = pd.read_excel(file)
    


    # Renommer des variables
    st.write("Renommer des variables :")
    cols_to_rename = st.multiselect("Sélectionnez les variables à renommer", df.columns)
    for col in cols_to_rename:
        new_col_name = st.text_input(f"Nom de la variable {col}", col)
        if new_col_name != col:
            df = df.rename(columns={col: new_col_name})

    # Créer une nouvelle variable
    st.write("Créer une nouvelle variable :")
    new_var_name = st.text_input("Nom de la nouvelle variable")
    if new_var_name:
        formula = st.text_input("Formule de la nouvelle variable (utilisez les noms des variables existantes)")
        if formula:
            try:
                new_var = eval(formula, {}, {"df": df})
                df[new_var_name] = new_var
            except:
                st.warning("La formule est incorrecte.")

    # Créer un nouveau DataFrame avec des variables sélectionnées
    st.write("Créer un nouveau DataFrame avec des variables sélectionnées :")
    new_df_name = st.text_input("Nom du nouveau DataFrame")
    selected_cols = st.multiselect("Sélectionnez les variables à inclure dans le nouveau DataFrame", df.columns)
    if selected_cols and new_df_name:
        new_df = df[selected_cols]
        st.write(f"Le nouveau DataFrame '{new_df_name}' a été créé avec les variables suivantes :")
        st.write(selected_cols)
        st.write(f"Les premières lignes du nouveau DataFrame {new_df_name} :" )
        st.write(new_df.head())


    # Affichage des premières lignes du fichier
    if analyse_options["Afficher les premières lignes"]:
        df = pd.read_excel(file)
        st.write("Les premières lignes du fichier :")
        st.write(df.head())

    # Affichage du nombre de lignes et de colonnes du fichier
    if analyse_options["Afficher le nombre de lignes et de colonnes"]:
        if "df" not in locals():
            df = pd.read_excel(file)
        st.write("Le nombre de lignes et de colonnes :")
        st.write(df.shape)

    # Affichage des statistiques descriptives du fichier
    if analyse_options["Afficher les statistiques descriptives"]:
        if "df" not in locals():
            df = pd.read_excel(file)
        st.write("Les statistiques descriptives :")
        st.write(df.describe())

    # Analyse de la matrice de corrélation
    if analyse_options["Analyser la matrice de corrélation"]:
        if "df" not in locals():
            df = pd.read_excel(file)
        corr = df.corr()
        st.write("La matrice de corrélation :")
        st.write(corr)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    # Analyse des variables catégorielles
    if analyse_options["Analyser les variables catégorielles"]:
        if "df" not in locals():
            df = pd.read_excel(file)
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            st.write("Analyse des variables catégorielles :")
            for col in categorical_cols:
                st.write(f"Variable {col}")
                st.write(df[col].value_counts())
                sns.countplot(x=col, data=df)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

    # Analyse des variables numériques
    if analyse_options["Analyser les variables numériques"]:
        if "df" not in locals():
            df = pd.read_excel(file)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numerical_cols:
            st.write("Analyse des variables numériques :")
            for col in numerical_cols:
                st.write(f"Variable {col}")
                sns.histplot(df[col], kde=True)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()





# Ajouter un bouton pour télécharger les résultats au format HTML
def get_table_download_link(new_df):
    # Convertir le DataFrame en HTML à l'aide de StringIO
    output = StringIO()
    new_df.to_html(output, index=False)
    # Encoder le fichier HTML en base64 pour le téléchargement
    b64 = base64.b64encode(output.getvalue().encode()).decode()
    # Générer un lien de téléchargement pour le fichier HTML
    href = f'<a href="data:text/html;base64,{b64}" download="resultats.html">Télécharger les résultats (HTML)</a>'
    return href

# Afficher le bouton de téléchargement dans Streamlit
if st.button("Télécharger les résultats (HTML)"):
    st.markdown(get_table_download_link(new_df), unsafe_allow_html=True)





# Ajouter un footer
st.sidebar.markdown("---")
st.sidebar.write("PAPA SEYDOU WANE || BDA-UNCHK")