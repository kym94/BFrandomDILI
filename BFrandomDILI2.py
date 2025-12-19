import streamlit as st
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
import os

# Suppression des avertissements
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Toxicité Hépatique (DILI)",
    page_icon="💊",
    layout="wide"
)

# CSS pour augmenter la taille des caractères
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    h1 {
        font-size: 2.5rem !important;
    }
    h2 {
        font-size: 2rem !important;
    }
    h3 {
        font-size: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation de session_state pour les exemples
if 'smiles' not in st.session_state:
    st.session_state.smiles = ""


# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle Random Forest depuis le fichier pickle"""
    # Chercher le modèle dans le dossier courant d'abord
    model_filename = 'best_model_20251026_200415.pkl'

    # Liste des chemins possibles
    possible_paths = [
        model_filename,  # Dossier courant
        os.path.join(os.path.dirname(__file__), model_filename),  # Dossier du script
        '/mnt/user-data/uploads/' + model_filename,  # Chemin original (pour Claude)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.success(f"✅ Modèle chargé depuis: {path}")
                return model
            except Exception as e:
                continue

    # Si aucun chemin ne fonctionne, afficher une erreur claire
    st.error(f"""
    ❌ **Modèle introuvable !**

    Le fichier `{model_filename}` n'a pas été trouvé.

    **Solution :**
    1. Téléchargez le fichier `{model_filename}`
    2. Placez-le dans le même dossier que `BFrandomDILI.py`
    3. Relancez l'application

    **Dossier actuel :** `{os.getcwd()}`
    """)
    st.stop()
    return None


# Fonction pour calculer les descripteurs physicochimiques
def calculate_descriptors(mol):
    """Calcule les 9 descripteurs physicochimiques à partir d'une molécule RDKit"""
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol)
    }
    return descriptors


# Fonction pour calculer les empreintes Morgan
def calculate_morgan_fingerprint(mol, radius=2, nBits=1024):
    """Calcule les empreintes moléculaires Morgan (rayon 2, 1024 bits)"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


# Fonction pour préparer les features complètes
def prepare_features(smiles):
    """Prépare toutes les features nécessaires pour la prédiction"""
    try:
        # Convertir SMILES en molécule RDKit
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None, "SMILES invalide. Veuillez vérifier la formule."

        # Calculer les descripteurs physicochimiques
        descriptors = calculate_descriptors(mol)

        # Calculer les empreintes Morgan
        morgan_fp = calculate_morgan_fingerprint(mol)

        # Créer un DataFrame avec les descriptors
        descriptor_df = pd.DataFrame([descriptors])

        # Créer un DataFrame pour les empreintes Morgan
        morgan_cols = [f'Morgan_R2_1024_bit_{i:04d}' for i in range(1024)]
        morgan_df = pd.DataFrame([morgan_fp], columns=morgan_cols)

        # Combiner les deux DataFrames
        features = pd.concat([descriptor_df, morgan_df], axis=1)

        return features, descriptors

    except Exception as e:
        return None, f"Erreur lors du calcul des descripteurs: {str(e)}"


# Fonction pour détecter la colonne SMILES
def detect_smiles_column(df):
    """Détecte automatiquement la colonne contenant les SMILES"""
    possible_names = ['smiles', 'SMILES', 'Smiles', 'smile', 'SMILE', 'Smile', 'SMILES_string', 'smiles_string']

    # Chercher une correspondance exacte
    for col in df.columns:
        if col in possible_names:
            return col

    # Chercher une correspondance partielle (insensible à la casse)
    for col in df.columns:
        if 'smiles' in col.lower():
            return col

    return None


# Fonction pour prédire un batch de molécules
def predict_batch(df, smiles_column, model):
    """Prédit la toxicité pour un DataFrame de molécules"""
    results = []

    for idx, row in df.iterrows():
        smiles = row[smiles_column]

        try:
            if pd.isna(smiles) or smiles == '':
                results.append({
                    'Prediction': 'N/A',
                    'Probabilite_Toxique (%)': np.nan,
                    'Probabilite_Non_Toxique (%)': np.nan,
                    'Statut': 'SMILES manquant'
                })
            else:
                features, descriptors = prepare_features(str(smiles))

                if features is None:
                    results.append({
                        'Prediction': 'Erreur',
                        'Probabilite_Toxique (%)': np.nan,
                        'Probabilite_Non_Toxique (%)': np.nan,
                        'Statut': 'SMILES invalide'
                    })
                else:
                    prediction = model.predict(features)[0]
                    prediction_proba = model.predict_proba(features)[0]

                    results.append({
                        'Prediction': 'TOXIQUE' if prediction == 1 else 'NON TOXIQUE',
                        'Probabilite_Toxique (%)': round(prediction_proba[1] * 100, 2),
                        'Probabilite_Non_Toxique (%)': round(prediction_proba[0] * 100, 2),
                        'Statut': 'OK'
                    })
        except Exception as e:
            results.append({
                'Prediction': 'Erreur',
                'Probabilite_Toxique (%)': np.nan,
                'Probabilite_Non_Toxique (%)': np.nan,
                'Statut': f'Erreur: {str(e)}'
            })

    results_df = pd.DataFrame(results)
    return pd.concat([df, results_df], axis=1)


# Fonction pour charger la chimiothèque
@st.cache_data
def load_chemolibrary():
    """Charge le fichier de chimiothèque"""
    try:
        # Noms possibles du fichier (avec espaces ou underscores)
        possible_filenames = [
            'molecules pharmacopee OOAS et proprites.xlsx',  # Nom avec espaces
            'molecules_pharmacopee_OOAS_et_proprites.xlsx',  # Nom avec underscores
        ]

        # Chemins possibles pour chaque nom
        possible_paths = []
        for filename in possible_filenames:
            possible_paths.extend([
                filename,  # Dossier courant
                os.path.join(os.path.dirname(__file__), filename),  # Dossier du script
                '/mnt/user-data/uploads/' + filename,  # Chemin original (pour Claude)
            ])

        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_excel(path)
                # Remplir les NaN dans Plante avec la valeur précédente (forward fill)
                df['Plante'] = df['Plante'].ffill()
                return df

        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement de la chimiothèque : {str(e)}")
        return None


# Fonction pour calculer les statistiques descriptives
def calculate_descriptive_stats(df, column):
    """Calcule les statistiques descriptives pour une colonne"""
    if df[column].dtype in ['object', 'category']:
        # Variables catégorielles : proportions
        counts = df[column].value_counts()
        proportions = (counts / len(df) * 100).round(2)
        stats = pd.DataFrame({
            'Catégorie': counts.index,
            'Fréquence': counts.values,
            'Proportion (%)': proportions.values
        })
        return stats, 'categorical'
    else:
        # Variables continues : statistiques
        stats = {
            'Moyenne': df[column].mean(),
            'Médiane': df[column].median(),
            'Écart-type': df[column].std(),
            'Minimum': df[column].min(),
            'Maximum': df[column].max(),
            'Q1 (25%)': df[column].quantile(0.25),
            'Q3 (75%)': df[column].quantile(0.75)
        }
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Valeur']
        return stats_df, 'continuous'


# Titre principal
st.title("💊 PRÉDICTION DE LA TOXICITÉ HÉPATIQUE")
st.markdown("---")

st.markdown("""
Cette application utilise un modèle de **Random Forest** pour prédire la toxicité hépatique 
(Drug-Induced Liver Injury - DILI) d'une molécule à partir de sa formule SMILES.
""")

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs(["🔍 PRÉDICTION UNIQUE", "📊 PRÉDICTION EN BATCH", "🌿 CHIMIOTHÈQUE", "ℹ️ À PROPOS"])

# ==================== ONGLET 1: PRÉDICTION UNIQUE ====================
with tab1:
    # Section d'entrée
    st.header("📝 Saisie de la molécule")

    # Zone de texte pour le SMILES
    smiles_input = st.text_input(
        "Entrez la formule SMILES de la molécule :",
        value=st.session_state.smiles,
        placeholder="Ex: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirine)",
        help="SMILES = Simplified Molecular Input Line Entry System"
    )

    # Mettre à jour session_state si l'utilisateur tape
    if smiles_input != st.session_state.smiles:
        st.session_state.smiles = smiles_input

    # Exemples de molécules
    st.markdown("**Exemples de molécules :**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Aspirine"):
            st.session_state.smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
            st.rerun()

    with col2:
        if st.button("Paracétamol"):
            st.session_state.smiles = "CC(=O)NC1=CC=C(C=C1)O"
            st.rerun()

    with col3:
        if st.button("Ibuprofène"):
            st.session_state.smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
            st.rerun()

    # Utiliser la valeur de session_state pour la prédiction
    smiles_to_predict = st.session_state.smiles

    # Bouton de prédiction
    if st.button("🔍 Prédire la toxicité", type="primary"):
        if not smiles_to_predict:
            st.warning("⚠️ Veuillez entrer une formule SMILES.")
        else:
            with st.spinner("Calcul en cours..."):
                # Préparer les features
                features, result = prepare_features(smiles_to_predict)

                if features is None:
                    st.error(f"❌ {result}")
                else:
                    # Charger le modèle
                    model = load_model()

                    # Faire la prédiction
                    prediction = model.predict(features)[0]
                    prediction_proba = model.predict_proba(features)[0]

                    # Afficher les résultats
                    st.markdown("---")
                    st.header("📊 Résultats de la prédiction")

                    # Résultat principal
                    col1, col2 = st.columns(2)

                    with col1:
                        if prediction == 1:
                            st.error("### ⚠️ TOXIQUE")
                            st.markdown("La molécule est **prédite comme toxique** pour le foie.")
                        else:
                            st.success("### ✅ NON TOXIQUE")
                            st.markdown("La molécule est **prédite comme non toxique** pour le foie.")

                    with col2:
                        st.metric(
                            label="Probabilité de toxicité",
                            value=f"{prediction_proba[1] * 100:.1f}%"
                        )
                        st.metric(
                            label="Probabilité de non-toxicité",
                            value=f"{prediction_proba[0] * 100:.1f}%"
                        )

                    # Afficher les descripteurs calculés
                    st.markdown("---")
                    st.subheader("🧪 Descripteurs physicochimiques calculés")

                    desc_df = pd.DataFrame([result]).T
                    desc_df.columns = ['Valeur']
                    desc_df.index.name = 'Descripteur'

                    st.dataframe(desc_df, use_container_width=True)

                    # Information sur le modèle
                    st.markdown("---")
                    st.info("""
                    **ℹ️ À propos du modèle :**
                    - Type : Random Forest
                    - Features : 9 descripteurs physicochimiques + 1024 bits d'empreintes Morgan (R=2)
                    - Classes : 0 = Non toxique, 1 = Toxique
                    """)

# ==================== ONGLET 2: PRÉDICTION EN BATCH ====================
with tab2:
    st.header("📊 Prédiction en Batch")

    st.markdown("""
    Chargez un fichier **Excel (.xlsx)** ou **CSV (.csv)** contenant une colonne avec les formules SMILES.
    L'application détectera automatiquement la colonne SMILES et ajoutera les prédictions.
    """)

    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier",
        type=['xlsx', 'csv'],
        help="Le fichier doit contenir une colonne 'SMILES' (ou similaire)"
    )

    if uploaded_file is not None:
        try:
            # Lecture du fichier
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Fichier chargé avec succès : **{uploaded_file.name}**")
            st.write(f"**Nombre de lignes :** {len(df)}")
            st.write(f"**Colonnes :** {', '.join(df.columns.tolist())}")

            # Détecter la colonne SMILES
            smiles_col = detect_smiles_column(df)

            if smiles_col is None:
                st.error("""
                ❌ **Colonne SMILES non détectée !**

                Veuillez vous assurer que votre fichier contient une colonne nommée 'SMILES' (ou similaire).
                """)

                st.info("**Colonnes disponibles :** " + ", ".join(df.columns.tolist()))

                # Option pour sélectionner manuellement
                smiles_col = st.selectbox(
                    "Sélectionnez manuellement la colonne contenant les SMILES :",
                    options=df.columns.tolist()
                )
            else:
                st.success(f"✅ Colonne SMILES détectée : **{smiles_col}**")

            # Aperçu des données
            st.subheader("📋 Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)

            # Bouton pour lancer les prédictions
            if st.button("🚀 Lancer les prédictions", type="primary", key="batch_predict"):
                with st.spinner(f"Prédiction en cours pour {len(df)} molécules..."):
                    # Charger le modèle
                    model = load_model()

                    # Prédictions en batch
                    df_results = predict_batch(df, smiles_col, model)

                    # Afficher les résultats
                    st.success("✅ Prédictions terminées !")

                    # Statistiques
                    st.subheader("📈 Statistiques")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        total = len(df_results)
                        st.metric("Total", total)

                    with col2:
                        toxiques = len(df_results[df_results['Prediction'] == 'TOXIQUE'])
                        st.metric("Toxiques", toxiques, delta=f"{toxiques / total * 100:.1f}%")

                    with col3:
                        non_toxiques = len(df_results[df_results['Prediction'] == 'NON TOXIQUE'])
                        st.metric("Non toxiques", non_toxiques, delta=f"{non_toxiques / total * 100:.1f}%")

                    with col4:
                        erreurs = len(df_results[df_results['Prediction'].isin(['Erreur', 'N/A'])])
                        st.metric("Erreurs", erreurs)

                    # Afficher les résultats
                    st.subheader("📊 Résultats détaillés")
                    st.dataframe(df_results, use_container_width=True)

                    # Téléchargement des résultats
                    st.subheader("💾 Télécharger les résultats")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Export CSV
                        csv = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv,
                            file_name=f"predictions_DILI_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Export Excel
                        from io import BytesIO

                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_results.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_data = output.getvalue()

                        st.download_button(
                            label="📥 Télécharger Excel",
                            data=excel_data,
                            file_name=f"predictions_DILI_{uploaded_file.name.split('.')[0]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
            st.info("Assurez-vous que le fichier est au bon format (CSV ou Excel).")

# ==================== ONGLET 3: CHIMIOTHÈQUE ====================
with tab3:
    st.header("🌿 Chimiothèque - Pharmacopée OOAS")

    st.markdown("""
    Explorez les **molécules de la pharmacopée OOAS** et leurs propriétés physicochimiques et pharmacologiques.
    Visualisez les statistiques descriptives par plante.
    """)

    # Charger la chimiothèque
    df_chemo = load_chemolibrary()

    if df_chemo is None:
        st.error("""
        ❌ **Fichier de chimiothèque non trouvé !**

        Le fichier de la pharmacopée OOAS est requis. Il peut avoir l'un de ces noms :
        - `molecules pharmacopee OOAS et proprites.xlsx` (avec espaces)
        - `molecules_pharmacopee_OOAS_et_proprites.xlsx` (avec underscores)

        Placez-le dans le même dossier que l'application.
        """)
    else:
        st.success(
            f"✅ Chimiothèque chargée : **{len(df_chemo)} molécules** de **{df_chemo['Plante'].nunique()} plantes**")

        # Statistiques générales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total molécules", len(df_chemo))
        with col2:
            st.metric("Plantes", df_chemo['Plante'].nunique())
        with col3:
            avg_mol = df_chemo.groupby('Plante').size().mean()
            st.metric("Moy. molécules/plante", f"{avg_mol:.1f}")

        st.markdown("---")

        # Sélection de la plante
        st.subheader("🔍 Sélection de plante(s)")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Option pour toutes les plantes ou sélection spécifique
            all_plants = st.checkbox("Afficher toutes les plantes", value=False)

            if not all_plants:
                selected_plants = st.multiselect(
                    "Sélectionnez une ou plusieurs plantes :",
                    options=sorted(df_chemo['Plante'].unique()),
                    default=[sorted(df_chemo['Plante'].unique())[0]]
                )
            else:
                selected_plants = df_chemo['Plante'].unique().tolist()

        with col2:
            if not all_plants and selected_plants:
                mol_count = df_chemo[df_chemo['Plante'].isin(selected_plants)].shape[0]
                st.metric("Molécules sélectionnées", mol_count)

        if selected_plants:
            # Filtrer les données
            df_filtered = df_chemo[df_chemo['Plante'].isin(selected_plants)]

            # Onglets pour différentes vues
            subtab1, subtab2, subtab3 = st.tabs([
                "📋 Données",
                "📊 Statistiques Physicochimiques",
                "💊 Statistiques Pharmacologiques"
            ])

            # ========== SOUS-ONGLET 1: Données ==========
            with subtab1:
                st.subheader("📋 Molécules sélectionnées")

                # Options d'affichage
                show_cols = st.multiselect(
                    "Colonnes à afficher :",
                    options=df_filtered.columns.tolist(),
                    default=['Plante', 'Molecule', 'Canonical SMILES', 'MW', 'TPSA', 'Consensus Log P']
                )

                if show_cols:
                    st.dataframe(df_filtered[show_cols], use_container_width=True, height=400)

                    # Export des données filtrées
                    csv = df_filtered[show_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les données (CSV)",
                        data=csv,
                        file_name=f"chimotheque_{'_'.join(selected_plants[:2])}.csv",
                        mime="text/csv"
                    )

            # ========== SOUS-ONGLET 2: Stats Physicochimiques ==========
            with subtab2:
                st.subheader("📊 Statistiques Physicochimiques")

                # Colonnes physicochimiques
                physico_cols = [
                    'MW', 'Heavy atoms', 'Aromatic heavy atoms', 'Fraction Csp3',
                    'Rotatable bonds', 'H-bond acceptors', 'H-bond donors', 'MR', 'TPSA',
                    'Consensus Log P', 'Bioavailability Score', 'Synthetic Accessibility'
                ]

                # Filtrer les colonnes qui existent
                physico_cols = [col for col in physico_cols if col in df_filtered.columns]

                # Sélection de la propriété
                selected_prop = st.selectbox(
                    "Sélectionnez une propriété physicochimique :",
                    options=physico_cols,
                    index=0
                )

                if selected_prop:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(f"**Statistiques pour : {selected_prop}**")
                        stats_df, stat_type = calculate_descriptive_stats(df_filtered, selected_prop)

                        if stat_type == 'continuous':
                            st.dataframe(stats_df, use_container_width=True)
                        else:
                            st.dataframe(stats_df, use_container_width=True)

                    with col2:
                        if stat_type == 'continuous':
                            # Histogramme
                            import plotly.express as px

                            fig = px.histogram(
                                df_filtered,
                                x=selected_prop,
                                nbins=20,
                                title=f"Distribution de {selected_prop}",
                                labels={selected_prop: selected_prop, 'count': 'Fréquence'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Graphique en barres pour catégories
                            import plotly.express as px

                            fig = px.bar(
                                stats_df,
                                x='Catégorie',
                                y='Fréquence',
                                title=f"Distribution de {selected_prop}",
                                text='Proportion (%)'
                            )
                            fig.update_traces(texttemplate='%{text}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)

                # Tableau récapitulatif de toutes les propriétés physicochimiques
                st.markdown("---")
                st.markdown("**📋 Récapitulatif de toutes les propriétés**")

                summary_data = []
                for col in physico_cols:
                    if df_filtered[col].dtype in ['int64', 'float64']:
                        summary_data.append({
                            'Propriété': col,
                            'Moyenne': f"{df_filtered[col].mean():.2f}",
                            'Médiane': f"{df_filtered[col].median():.2f}",
                            'Écart-type': f"{df_filtered[col].std():.2f}",
                            'Min': f"{df_filtered[col].min():.2f}",
                            'Max': f"{df_filtered[col].max():.2f}"
                        })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)

            # ========== SOUS-ONGLET 3: Stats Pharmacologiques ==========
            with subtab3:
                st.subheader("💊 Statistiques Pharmacologiques")

                # Colonnes pharmacologiques
                pharmaco_cols = [
                    'GI absorption', 'BBB permeant', 'Pgp substrate',
                    'CYP1A2 inhibitor', 'CYP2C19 inhibitor', 'CYP2C9 inhibitor',
                    'CYP2D6 inhibitor', 'CYP3A4 inhibitor',
                    'Lipinski #violations', 'PAINS alerts', 'Brenk alerts'
                ]

                # Filtrer les colonnes qui existent
                pharmaco_cols = [col for col in pharmaco_cols if col in df_filtered.columns]

                # Sélection de la propriété
                selected_pharmaco = st.selectbox(
                    "Sélectionnez une propriété pharmacologique :",
                    options=pharmaco_cols,
                    index=0
                )

                if selected_pharmaco:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(f"**Statistiques pour : {selected_pharmaco}**")
                        stats_df, stat_type = calculate_descriptive_stats(df_filtered, selected_pharmaco)
                        st.dataframe(stats_df, use_container_width=True)

                    with col2:
                        if stat_type == 'continuous':
                            # Histogramme
                            import plotly.express as px

                            fig = px.histogram(
                                df_filtered,
                                x=selected_pharmaco,
                                nbins=20,
                                title=f"Distribution de {selected_pharmaco}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Graphique en barres
                            import plotly.express as px

                            fig = px.bar(
                                stats_df,
                                x='Catégorie',
                                y='Fréquence',
                                title=f"Distribution de {selected_pharmaco}",
                                text='Proportion (%)'
                            )
                            fig.update_traces(texttemplate='%{text}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)

                # Récapitulatif des propriétés catégorielles
                st.markdown("---")
                st.markdown("**📋 Récapitulatif des propriétés catégorielles**")

                cat_cols = [col for col in pharmaco_cols if df_filtered[col].dtype == 'object']

                if cat_cols:
                    for col in cat_cols:
                        with st.expander(f"📊 {col}"):
                            stats_df, _ = calculate_descriptive_stats(df_filtered, col)

                            col_a, col_b = st.columns([1, 2])
                            with col_a:
                                st.dataframe(stats_df, use_container_width=True)
                            with col_b:
                                import plotly.express as px

                                fig = px.pie(
                                    stats_df,
                                    values='Fréquence',
                                    names='Catégorie',
                                    title=f"Distribution de {col}"
                                )
                                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Veuillez sélectionner au moins une plante.")

# ==================== ONGLET 4: À PROPOS ====================
with tab4:
    st.header("ℹ️ À Propos du Modèle")

    # Avertissement éthique
    st.error("""
    ⚠️ **AVERTISSEMENT ÉTHIQUE ET LÉGAL**

    Cette application est développée **À DES FINS DE RECHERCHE UNIQUEMENT**.

    Elle **N'EST PAS destinée à un usage clinique** et ne doit pas être utilisée pour :
    - Prendre des décisions thérapeutiques
    - Diagnostiquer des patients
    - Remplacer l'avis d'un professionnel de santé

    Les prédictions fournies par ce modèle sont des estimations basées sur des données d'entraînement 
    et ne constituent pas une garantie de toxicité ou de sécurité d'une molécule.

    **Toute utilisation clinique nécessite une validation réglementaire appropriée.**
    """)

    st.markdown("---")

    # Section 1: Méthode de développement
    st.subheader("🔬 Méthode de Développement")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Algorithme :**
        - Random Forest (Forêt Aléatoire)
        - Ensemble de 100 arbres de décision

        **Stratégie de features :**
        - **9 descripteurs physicochimiques** :
          - Poids moléculaire (MolWt)
          - LogP (MolLogP)
          - Accepteurs H (NumHAcceptors)
          - Donneurs H (NumHDonors)
          - Surface de Labute (LabuteASA)
          - Surface polaire (TPSA)
          - Liaisons rotatives (NumRotatableBonds)
          - Nombre de cycles (RingCount)
          - Cycles aromatiques (NumAromaticRings)

        - **1024 bits d'empreintes Morgan** (rayon = 2)
        - **Total : 1033 features**
        """)

    with col2:
        st.markdown("""
        **Jeux de données :**
        - **Entraînement** : 966 molécules
          - Classe 0 (Non toxique) : 376
          - Classe 1 (Toxique) : 590

        - **Test interne** : 244 molécules
          - Classe 0 : 91
          - Classe 1 : 153

        - **Validation externe** : 156 molécules
          - Classe 0 : 65
          - Classe 1 : 91

        **Validation :**
        - Cross-validation 5-fold stratifiée
        - Test interne
        - Validation externe indépendante
        """)

    st.markdown("---")

    # Section 2: Grille d'hyperparamètres
    st.subheader("⚙️ Grille d'Hyperparamètres")

    st.markdown("""
    Le modèle Random Forest a été optimisé par recherche en grille (GridSearchCV) avec les paramètres suivants :
    """)

    hyperparam_data = {
        'Hyperparamètre': [
            'n_estimators',
            'max_depth',
            'min_samples_split',
            'min_samples_leaf',
            'max_features',
            'class_weight',
            'random_state'
        ],
        'Valeurs testées': [
            '[50, 100, 200]',
            '[10, 20, 30, None]',
            '[2, 5, 10]',
            '[1, 2, 4]',
            '["sqrt", "log2"]',
            '["balanced", None]',
            '42'
        ],
        'Valeur optimale': [
            '100',
            'None',
            '2',
            '1',
            'sqrt',
            'balanced',
            '42'
        ]
    }

    df_hyperparam = pd.DataFrame(hyperparam_data)
    st.dataframe(df_hyperparam, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 3: Métriques d'évaluation
    st.subheader("📊 Métriques d'Évaluation du Modèle")

    st.markdown("""
    **Performance sur l'ensemble de validation externe** (le plus représentatif des performances réelles) :
    """)

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "76.28%")
        st.metric("Balanced Accuracy", "76.15%")

    with col2:
        st.metric("Sensibilité (Recall)", "76.92%")
        st.metric("Spécificité", "75.38%")

    with col3:
        st.metric("Précision", "81.40%")
        st.metric("F1 Score", "79.10%")

    with col4:
        st.metric("AUC-ROC", "0.834")
        st.metric("AUC-PR", "0.863")

    st.markdown("---")

    # Tableau complet des métriques
    st.markdown("**📈 Métriques détaillées**")

    metrics_data = {
        'Métrique': [
            'Accuracy',
            'Balanced Accuracy',
            'Sensibilité (Recall / True Positive Rate)',
            'Spécificité (True Negative Rate)',
            'Précision (Precision / PPV)',
            'F1 Score',
            'F1 Macro',
            'AUC-ROC',
            'AUC-PR',
            'MCC (Matthews Correlation Coefficient)',
            'Cohen\'s Kappa',
            'G-mean',
            'NPV (Negative Predictive Value)'
        ],
        'Validation Externe': [
            '0.7628',
            '0.7615',
            '0.7692',
            '0.7538',
            '0.8140',
            '0.7910',
            '0.7584',
            '0.8340',
            '0.8629',
            '0.5185',
            '0.5174',
            '0.7615',
            '0.7000'
        ],
        'Cross-Validation (5-fold)': [
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            '0.6644 ± 0.0511',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-'
        ]
    }

    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 4: Matrice de confusion
    st.subheader("🎯 Matrice de Confusion (Validation Externe)")

    col1, col2 = st.columns([1, 2])

    with col1:
        confusion_data = {
            '': ['Prédit Négatif', 'Prédit Positif'],
            'Réel Négatif': ['49 (TN)', '16 (FP)'],
            'Réel Positif': ['21 (FN)', '70 (TP)']
        }
        df_confusion = pd.DataFrame(confusion_data)
        st.dataframe(df_confusion, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("""
        **Interprétation :**

        - **Vrais Négatifs (TN)** : 49 molécules correctement prédites comme non toxiques
        - **Vrais Positifs (TP)** : 70 molécules correctement prédites comme toxiques
        - **Faux Positifs (FP)** : 16 molécules non toxiques prédites comme toxiques
        - **Faux Négatifs (FN)** : 21 molécules toxiques prédites comme non toxiques

        **Taux de réussite :** 119/156 = 76.28%
        """)

    st.markdown("---")

    # Section 5: Interprétation des métriques
    st.subheader("📖 Interprétation des Métriques Clés")

    with st.expander("🔍 Cliquez pour voir les définitions"):
        st.markdown("""
        **Sensibilité (76.92%)** : Capacité du modèle à identifier correctement les molécules toxiques.
        - Sur 91 molécules toxiques, le modèle en détecte correctement 70.

        **Spécificité (75.38%)** : Capacité du modèle à identifier correctement les molécules non toxiques.
        - Sur 65 molécules non toxiques, le modèle en identifie correctement 49.

        **Accuracy (76.28%)** : Proportion totale de prédictions correctes.
        - Le modèle fait des prédictions correctes dans 76.28% des cas.

        **Balanced Accuracy (76.15%)** : Moyenne de la sensibilité et de la spécificité.
        - Prend en compte le déséquilibre des classes.

        **AUC-ROC (0.834)** : Aire sous la courbe ROC.
        - Mesure la capacité du modèle à discriminer entre les deux classes.
        - Valeur proche de 1 = excellent, proche de 0.5 = aléatoire.

        **AUC-PR (0.863)** : Aire sous la courbe Precision-Recall.
        - Particulièrement pertinente pour les datasets déséquilibrés.

        **F1 Score (79.10%)** : Moyenne harmonique de la précision et du recall.
        - Équilibre entre la précision et la sensibilité.

        **MCC (0.5185)** : Coefficient de corrélation de Matthews.
        - Mesure de la qualité globale des prédictions (-1 à +1).
        - Valeur > 0.5 indique une bonne performance.
        """)

    st.markdown("---")

    # Section 6: Recommandations
    st.subheader("💡 Recommandations d'Utilisation")

    st.warning("""
    **Points d'attention :**

    1. **Domaine d'applicabilité** : Le modèle est entraîné sur des molécules de type médicament. 
       Les prédictions pour des molécules très différentes peuvent être moins fiables.

    2. **Interprétation des résultats** : Une prédiction "toxique" indique un risque potentiel 
       qui nécessite des investigations supplémentaires, pas une certitude absolique.

    3. **Faux négatifs** : Le modèle peut manquer ~23% des molécules toxiques (FN).
       Ne pas se fier uniquement à ce modèle pour écarter un risque de toxicité.

    4. **Usage en recherche** : Utiliser ce modèle comme outil de criblage initial 
       dans un pipeline de découverte de médicaments, pas comme décision finale.

    5. **Validation expérimentale** : Toute prédiction doit être validée par des tests 
       in vitro et in vivo avant toute application.
    """)

    st.markdown("---")

    # Section 7: Informations techniques
    st.subheader("🔧 Informations Techniques")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Frameworks utilisés :**
        - Python 3.13
        - Streamlit 1.50+
        - RDKit 2025.9.1
        - scikit-learn 1.7.0
        - Pandas 2.3.0
        - NumPy 1.23+
        - Plotly 5.0+
        """)

    with col2:
        st.markdown("""
        **Fichiers du modèle :**
        - Modèle : `best_model_20251026_200415.pkl`
        - Taille : 6.8 MB
        - Date d'entraînement : 26 octobre 2025
        - Random State : 42

        **Chimiothèque :**
        - 191 molécules
        - 30 plantes médicinales (Pharmacopée OOAS)
        """)

    st.markdown("---")

    # Footer
    st.info("""
    **📚 Références et Citation**

    Si vous utilisez cette application dans vos recherches, veuillez citer :
    - Nom du projet : Prédiction de Toxicité Hépatique (DILI)
    - Version : 1.2
    - Date : Octobre 2025

    Pour plus d'informations ou signaler des problèmes, veuillez contacter l'équipe de développement.
    """)

# Section d'information
with st.sidebar:
    st.header("ℹ️ Information")

    st.markdown("""
    ### Modules disponibles :

    **🔍 Prédiction Unique :**
    - Entrez un SMILES manuellement
    - Utilisation des exemples
    - Résultats détaillés instantanés

    **📊 Prédiction en Batch :**
    - Upload de fichier Excel/CSV
    - Détection automatique colonne SMILES
    - Prédictions multiples
    - Export des résultats

    **🌿 Chimiothèque :**
    - Pharmacopée OOAS
    - Exploration par plante
    - Statistiques descriptives
    - Visualisations interactives

    **ℹ️ À Propos :**
    - Méthode de développement
    - Grille d'hyperparamètres
    - Métriques d'évaluation
    - Avertissement éthique

    ---

    ### Descripteurs utilisés :

    **Physicochimiques (9) :**
    - Poids moléculaire (MolWt)
    - LogP (MolLogP)
    - Accepteurs H (NumHAcceptors)
    - Donneurs H (NumHDonors)
    - Surface de Labute (LabuteASA)
    - Surface polaire (TPSA)
    - Liaisons rotatives (NumRotatableBonds)
    - Nombre de cycles (RingCount)
    - Cycles aromatiques (NumAromaticRings)

    **Empreintes moléculaires :**
    - Morgan R2 (1024 bits)
    """)

    st.markdown("---")
    st.markdown("**Développé avec :** Streamlit + RDKit + scikit-learn")