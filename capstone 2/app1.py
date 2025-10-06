import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.title("Fuzzy Logic Based FMEA in Network Device Failure")

# ---------------------------
# 1. Load CSV
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Compute traditional RPN rank if column exists
    if 'RPN' in data.columns and 'RPN Rank' not in data.columns:
        data['RPN Rank'] = data['RPN'].rank(ascending=False, method='min').astype(int)
    
    # Store data in session_state if not already
    if 'data' not in st.session_state:
        st.session_state['data'] = data

    # ---------------------------
    # 2. Clustering
    # ---------------------------
    sod_data = st.session_state['data'][['Severity', 'Occurrence', 'Detection']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sod_data)

    n_clusters = 3

    # Run clustering only once
    if 'cluster_done' not in st.session_state or st.session_state.get('cluster_done')==False:
        if st.button("Run Clustering"):
            cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                X_scaled.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
            )
            cluster_labels = np.argmax(u, axis=0)
            st.session_state['data']['Cluster'] = cluster_labels
            st.session_state['cluster_done'] = True
            st.session_state['cluster_labels'] = cluster_labels
            st.session_state['fpc'] = fpc
            st.session_state['sil_score'] = silhouette_score(X_scaled, cluster_labels)
            st.success("✅ Clustering Completed")
            st.write(f"Silhouette Score: {round(st.session_state['sil_score'],3)}")
            st.write(f"Fuzzy Partition Coefficient (FPC): {round(st.session_state['fpc'],3)}")
            
            # PCA scatter plot
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            fig = plt.figure(figsize=(8,6))
            plt.scatter(pcs[:,0], pcs[:,1], c=cluster_labels, cmap='viridis', s=80, alpha=0.7)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Cluster Visualization (PCA Scatter Plot)")
            plt.tight_layout()
            st.pyplot(fig)
    elif st.session_state['cluster_done']:
        st.success("✅ Clustering Already Done")
        st.write(f"Silhouette Score: {round(st.session_state['sil_score'],3)}")
        st.write(f"Fuzzy Partition Coefficient (FPC): {round(st.session_state['fpc'],3)}")

    # ---------------------------
    # 3. Fuzzy Inference System (FIS)
    # ---------------------------
    if st.session_state.get('cluster_done', False):
        if 'fis_done' not in st.session_state or st.session_state.get('fis_done')==False:
            if st.button("Run Fuzzy Inference System"):
                data = st.session_state['data']

                RPN_min, RPN_max = data['RPN'].min(), data['RPN'].max()

                # Antecedents
                severity = ctrl.Antecedent(np.arange(0, 11, 1), 'Severity')
                occurrence = ctrl.Antecedent(np.arange(0, 11, 1), 'Occurrence')
                detection = ctrl.Antecedent(np.arange(0, 11, 1), 'Detection')
                rpn = ctrl.Consequent(np.arange(RPN_min, RPN_max+1, 1), 'RPN')

                # Membership functions
                severity['SL']  = fuzz.trimf(severity.universe, [0, 0, 2])
                severity['SMI'] = fuzz.trimf(severity.universe, [2, 3, 5])
                severity['SM']  = fuzz.trimf(severity.universe, [4, 6, 8])
                severity['SVH'] = fuzz.trimf(severity.universe, [7, 9, 9])
                severity['SHA'] = fuzz.trimf(severity.universe, [9, 10, 10])

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

                # Rules (example)
                rules = []

                 # SL + OR
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OR'] & detection['DAC'], rpn['RMI'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OR'] & detection['DH'],  rpn['RMI'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OR'] & detection['DM'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OR'] & detection['DL'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OR'] & detection['DAI'], rpn['RM'])),

                # SL + OVU
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OVU'] & detection['DAC'], rpn['RMI'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OVU'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OVU'] & detection['DM'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OVU'] & detection['DL'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OVU'] & detection['DAI'], rpn['RH'])),

# SL + OO
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OO'] & detection['DAC'], rpn['RMI'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OO'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OO'] & detection['DM'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OO'] & detection['DL'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OO'] & detection['DAI'], rpn['RH'])),

# SL + OP
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OP'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OP'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OP'] & detection['DM'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OP'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OP'] & detection['DAI'], rpn['RH'])),

# SL + OF
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OF'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OF'] & detection['DH'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OF'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OF'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SL'] & occurrence['OF'] & detection['DAI'], rpn['RE'])),


# SMI + OR
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OR'] & detection['DAC'], rpn['RMI'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OR'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OR'] & detection['DM'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OR'] & detection['DL'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OR'] & detection['DAI'], rpn['RM'])),

# SMI + OVU
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OVU'] & detection['DAC'], rpn['RMI'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OVU'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OVU'] & detection['DM'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OVU'] & detection['DL'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OVU'] & detection['DAI'], rpn['RH'])),

# SMI + OO
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OO'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OO'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OO'] & detection['DM'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OO'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OO'] & detection['DAI'], rpn['RH'])),

# SMI + OP
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OP'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OP'] & detection['DH'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OP'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OP'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OP'] & detection['DAI'], rpn['RE'])),

# SMI + OF
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OF'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OF'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OF'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OF'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SMI'] & occurrence['OF'] & detection['DAI'], rpn['RE'])),


                # SM + OR
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OR'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OR'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OR'] & detection['DM'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OR'] & detection['DL'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OR'] & detection['DAI'], rpn['RH'])),

                # SM + OVU
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OVU'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OVU'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OVU'] & detection['DM'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OVU'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OVU'] & detection['DAI'], rpn['RH'])),

                # SM + OO
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OO'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OO'] & detection['DH'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OO'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OO'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OO'] & detection['DAI'], rpn['RE'])),

                # SM + OP
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OP'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OP'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OP'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OP'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OP'] & detection['DAI'], rpn['RE'])),

                # SM + OF
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OF'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OF'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OF'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OF'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SM'] & occurrence['OF'] & detection['DAI'], rpn['RE'])),


                # SVH + OR
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OR'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OR'] & detection['DH'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OR'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OR'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OR'] & detection['DAI'], rpn['RE'])),

                # SVH + OVU
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OVU'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OVU'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OVU'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OVU'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OVU'] & detection['DAI'], rpn['RE'])),

                # SVH + OO
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OO'] & detection['DAC'], rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OO'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OO'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OO'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OO'] & detection['DAI'], rpn['RE'])),

                # SVH + OP
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OP'] & detection['DAC'], rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OP'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OP'] & detection['DM'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OP'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OP'] & detection['DAI'], rpn['RE'])),

                # SVH + OF
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OF'] & detection['DAC'], rpn['RH'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OF'] & detection['DH'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OF'] & detection['DM'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OF'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SVH'] & occurrence['OF'] & detection['DAI'], rpn['RE'])),



                # SHA + OR
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OR'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OR'] & detection['DH'],  rpn['RL'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OR'] & detection['DM'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OR'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OR'] & detection['DAI'], rpn['RH'])),

                # SHA + OVU
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OVU'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OVU'] & detection['DH'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OVU'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OVU'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OVU'] & detection['DAI'], rpn['RE'])),

                # SHA + OO
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OO'] & detection['DAC'], rpn['RL'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OO'] & detection['DH'],  rpn['RM'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OO'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OO'] & detection['DL'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OO'] & detection['DAI'], rpn['RE'])),

                # SHA + OP
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OP'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OP'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OP'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OP'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OP'] & detection['DAI'], rpn['RE'])),

                # SHA + OF
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OF'] & detection['DAC'], rpn['RM'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OF'] & detection['DH'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OF'] & detection['DM'],  rpn['RH'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OF'] & detection['DL'],  rpn['RE'])),
                rules.append(ctrl.Rule(severity['SHA'] & occurrence['OF'] & detection['DAI'], rpn['RE']))                   
                

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
                st.session_state['data'] = data
                st.session_state['fis_done'] = True
                st.success("✅ FIS RPN Computed Successfully")
                st.dataframe(data)
        elif st.session_state.get('fis_done', False):
            st.success("✅ FIS Already Computed")
            st.dataframe(st.session_state['data'])

    # ---------------------------
    # 4. Cluster Data Exploration & Bar Plot
    # ---------------------------
    data = st.session_state['data']
    if 'Cluster' in data.columns:
        selected_cluster = st.selectbox("Select Cluster to visualize", sorted(data['Cluster'].unique()))
        df_cluster = data[data['Cluster'] == selected_cluster].reset_index(drop=True)
        st.subheader(f"Cluster {selected_cluster} Data")
        st.dataframe(df_cluster)

        if "Fuzzy Rank" in df_cluster.columns:
            fig = plt.figure(figsize=(10,6))
            bar_width = 0.35
            x = np.arange(len(df_cluster))
            plt.bar(x, df_cluster["RPN Rank"], width=bar_width, label="Traditional RPN Rank", alpha=0.7)
            plt.bar(x + bar_width, df_cluster["Fuzzy Rank"], width=bar_width, label="Fuzzy RPN Rank", alpha=0.7)
            plt.xticks(x + bar_width/2, df_cluster["Failure Mode"], rotation=30, ha="right")
            plt.ylabel("Rank")
            plt.title(f"Cluster {selected_cluster} - Traditional vs Fuzzy RPN Rank")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
