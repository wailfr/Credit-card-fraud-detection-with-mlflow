import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import joblib  # Pour charger les modèles sauvegardés
import seaborn as sns  # Pour la visualisation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # type: ignore

# Charger les modèles sauvegardés avec joblib
models = {
    "SVM": joblib.load('models/SVM.pkl'),
    "XGBoost": joblib.load('models/XGBoost.pkl'),
    "Random Forest": joblib.load('models/RandomForest.pkl'),
    "Logistic Regression": joblib.load('models/LogisticRegression.pkl'),
    "Naïve Bayes": joblib.load('models/NaiveBayes.pkl')
}

# Fonction de prétraitement des données
def preprocess_data(df):
    df = df.drop(columns=["Time"])  # Suppression de la colonne Time
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])  # Normalisation
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Application de SMOTE pour équilibrer les classes
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled, X_test, y_test

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraude", "Fraude"], yticklabels=["Non-Fraude", "Fraude"])
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    st.pyplot(plt)

# Fonction principale de l'application Streamlit
def main():
    # Ajout du titre
    st.set_page_config(page_title="Détection de Fraude", layout="wide")
    st.title("🛡️ Détection de Fraude par Carte de Crédit")

    # Barre latérale pour la sélection du modèle
    st.sidebar.header("⚙️ Paramètres")
    model_choice = st.sidebar.selectbox("Choisissez un modèle", list(models.keys()))

    # Chargement du fichier CSV
    uploaded_file = st.file_uploader("📂 Téléchargez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu des données téléchargées :")
        st.dataframe(df.head())

        # Prétraitement des données
        X_train_resampled, y_train_resampled, X_test, y_test = preprocess_data(df)

        # Prédiction avec le modèle sélectionné
        model = models[model_choice]
        if st.button("🔍 Prédire"):
            y_pred = model.predict(X_test)
            
            # Afficher les résultats
            st.subheader("📊 Résultats de la Prédiction")
            plot_confusion_matrix(y_test, y_pred)  # Matrice de confusion
            
            # Affichage du rapport de classification
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write("### 📋 Rapport de Classification")
            st.json(report)

            # Affichage des transactions frauduleuses détectées
            fraudulent_transactions = X_test[y_pred == 1]
            if not fraudulent_transactions.empty:
                st.subheader("⚠️ Transactions Frauduleuses Détectées")
                st.dataframe(fraudulent_transactions)
            else:
                st.success("✅ Aucune transaction frauduleuse détectée.")

            # Graphique des transactions normales vs frauduleuses
            st.subheader("📈 Répartition des Transactions")
            fig, ax = plt.subplots()
            sns.countplot(x=y_pred, palette="coolwarm", ax=ax)
            ax.set_xticklabels(["Non-Fraude", "Fraude"])
            ax.set_title("Distribution des Prédictions")
            st.pyplot(fig)

# Lancer l'application Streamlit
if __name__ == "__main__":
    main()
