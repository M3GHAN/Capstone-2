import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ---------------------------
# 1. Load data
# ---------------------------
st.title("FMEA Analysis with Fuzzy RPN and Clustering")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # Add Traditional RPN Rank
    data['RPN Rank'] = data['RPN'].rank(ascending=False, method='min').astype(int)

    # Prepare SOD columns
    sod_data = data[['Severity', 'Occurrence', 'Detection']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sod_data)
    X_scaled_T = X_scaled.T

    # ---------------------------
    # 2. Run Clustering Button
    # ---------------------------
    n_clusters = st.slider("Select number of clusters", 2, 5, 3)

    if st.button("Run Clustering"):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled_T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
        )

        cluster_labels = np.argmax(u, axis=0)
        data['Cluster'] = cluster_labels

        sil_score = silhouette_score(X_scaled, cluster_labels)
        st.success(f"✅ Clustering Completed")
        st.write(f"Silhouette Score: {round(sil_score,3)}")
        st.write(f"Fuzzy Partition Coefficient (FPC): {round(fpc,3)}")

        # Scatter plot using PCA
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8,6))
        plt.scatter(pcs[:,0], pcs[:,1], c=cluster_labels, cmap='viridis', s=80, alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Cluster Visualization (PCA Scatter Plot)")
        st.pyplot()

    # ---------------------------
    # 3. Run Fuzzy Inference System (FIS)
    # ---------------------------
    if st.button("Run Fuzzy Inference System"):
        RPN_min, RPN_max = data['RPN'].min(), data['RPN'].max()

        # Antecedents
        severity = ctrl.Antecedent(np.arange(0, 11, 1), 'Severity')
        occurrence = ctrl.Antecedent(np.arange(0, 11, 1), 'Occurrence')
        detection = ctrl.Antecedent(np.arange(0, 11, 1), 'Detection')
        rpn = ctrl.Consequent(np.arange(RPN_min, RPN_max+1, 1), 'RPN')

        # Membership functions
        severity['SL']  = fuzz.trimf(severity.universe, [0,0,2])
        severity['SMI'] = fuzz.trimf(severity.universe, [2,3,5])
        severity['SM']  = fuzz.trimf(severity.universe, [4.5,6,8])
        severity['SVH'] = fuzz.trimf(severity.universe, [7,8.5,9])
        severity['SHA'] = fuzz.trimf(severity.universe, [9,10,10])

        occurrence['OR']  = fuzz.trimf(occurrence.universe, [0,0,2])
        occurrence['OVU'] = fuzz.trimf(occurrence.universe, [1,3,5])
        occurrence['OO']  = fuzz.trimf(occurrence.universe, [4,6,8])
        occurrence['OP']  = fuzz.trimf(occurrence.universe, [7,9,10])
        occurrence['OF']  = fuzz.trimf(occurrence.universe, [8,10,10])

        detection['DAC'] = fuzz.trimf(detection.universe, [0,0,2])
        detection['DH']  = fuzz.trimf(detection.universe, [1,3,5])
        detection['DM']  = fuzz.trimf(detection.universe, [4.5,6,7])
        detection['DL']  = fuzz.trimf(detection.universe, [6,8.5,9])
        detection['DAI'] = fuzz.trimf(detection.universe, [8,10,10])

        # Adaptive RPN membership
        rpn['RL']  = fuzz.trimf(rpn.universe, [RPN_min, RPN_min, RPN_min + 0.3*(RPN_max-RPN_min)])
        rpn['RMI'] = fuzz.trimf(rpn.universe, [RPN_min + 0.2*(RPN_max-RPN_min),
                                               RPN_min + 0.35*(RPN_max-RPN_min),
                                               RPN_min + 0.5*(RPN_max-RPN_min)])
        rpn['RM']  = fuzz.trimf(rpn.universe, [RPN_min + 0.4*(RPN_max-RPN_min),
                                               RPN_min + 0.55*(RPN_max-RPN_min),
                                               RPN_min + 0.7*(RPN_max-RPN_min)])
        rpn['RH']  = fuzz.trimf(rpn.universe, [RPN_min + 0.6*(RPN_max-RPN_min),
                                               RPN_min + 0.75*(RPN_max-RPN_min),
                                               RPN_min + 0.9*(RPN_max-RPN_min)])
        rpn['RE']  = fuzz.trimf(rpn.universe, [RPN_min + 0.8*(RPN_max-RPN_min),
                                               RPN_max, RPN_max])
        rpn.defuzzify_method = 'mom'

        # Example Rules
        rules = [ctrl.Rule(severity['SL'] & occurrence['OR'] & detection['DAC'], rpn['RL']),
                 ctrl.Rule(severity['SHA'] & occurrence['OF'] & detection['DAI'], rpn['RE'])]

        rpn_ctrl = ctrl.ControlSystem(rules)

        fis_outputs = []
        for _, row in data.iterrows():
            sim = ctrl.ControlSystemSimulation(rpn_ctrl)
            try:
                sim.input['Severity'] = row['Severity']
                sim.input['Occurrence'] = row['Occurrence']
                sim.input['Detection'] = row['Detection']
                sim.compute()
                fis_outputs.append(round(sim.output['RPN'], 3))
            except:
                fis_outputs.append(None)

        data['FIS_RPN'] = fis_outputs
        data['Fuzzy Rank'] = data['FIS_RPN'].rank(ascending=False, method='min').astype(int)

        st.success("✅ FIS RPN Computed Successfully")
        st.dataframe(data)

        # ---------------------------
        # 5. Fuzzy Membership Visualization
        # ---------------------------
        if st.checkbox("Show Fuzzy Membership Functions"):
            st.subheader("Severity Membership Functions")
            severity.view(sim=None)
            st.pyplot()

            st.subheader("Occurrence Membership Functions")
            occurrence.view(sim=None)
            st.pyplot()

            st.subheader("Detection Membership Functions")
            detection.view(sim=None)
            st.pyplot()

            st.subheader("RPN Output Membership Functions")
            rpn.view(sim=None)
            st.pyplot()

    # ---------------------------
    # 6. Cluster Data Exploration
    # ---------------------------
    if "Cluster" in data.columns:
        selected_cluster = st.selectbox("Select Cluster to visualize", sorted(data['Cluster'].unique()))
        df_cluster = data[data['Cluster'] == selected_cluster].reset_index(drop=True)

        st.subheader(f"Cluster {selected_cluster} Data")
        st.dataframe(df_cluster)

        # Bar plot ranks comparison
        if "Fuzzy Rank" in df_cluster.columns:
            plt.figure(figsize=(10,6))
            bar_width = 0.35
            x = range(len(df_cluster))
            plt.bar(x, df_cluster["RPN Rank"], width=bar_width, label="Traditional RPN Rank", alpha=0.7)
            plt.bar([p + bar_width for p in x], df_cluster["Fuzzy Rank"], width=bar_width, label="Fuzzy RPN Rank", alpha=0.7)
            plt.xticks([p + bar_width/2 for p in x], df_cluster["Failure Mode"], rotation=30, ha="right")
            plt.ylabel("Rank")
            plt.title(f"Cluster {selected_cluster} - Traditional vs Fuzzy RPN Rank")
            plt.legend()
            st.pyplot()
