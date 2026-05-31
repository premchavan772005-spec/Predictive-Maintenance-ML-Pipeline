import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predictive Maintenance ML Platform", layout="wide", page_icon="🏭")

st.title("🏭 Predictive Maintenance — Live ML Inference Platform")
st.markdown("**Real-time machine failure prediction using Random Forest Ensemble | IoT Telemetry → Risk Score → Action**")

@st.cache_data
def load_and_train():
    df = pd.read_csv("predictive_maintenance.csv")
    df.columns = df.columns.str.strip()

    le = LabelEncoder()
    df['Type_Encoded'] = le.fit_transform(df['Type'])

    features = ['Type_Encoded', 'Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Target'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    return model, df, le, features, precision, f1, cm, feature_imp

with st.spinner("⚙️ Training Random Forest model on 10,000 IoT telemetry records..."):
    model, df, le, features, precision, f1, cm, feature_imp = load_and_train()

tab1, tab2, tab3 = st.tabs([
    "🔴 Live Failure Predictor",
    "📊 Fleet Analytics Dashboard",
    "🧠 Model Performance"
])

with tab1:
    st.subheader("🔴 Real-Time Machine Failure Risk Predictor")
    st.markdown("Adjust the sensor readings below — the model predicts failure probability **instantly**")

    col_info, col_pred = st.columns([1.2, 1])

    with col_info:
        st.markdown("#### ⚙️ Machine Sensor Input Panel")

        machine_type = st.selectbox(
            "Machine Type",
            options=['L', 'M', 'H'],
            help="L = Low quality, M = Medium quality, H = High quality"
        )

        air_temp = st.slider(
            "🌡️ Air Temperature (K)",
            min_value=295.0, max_value=305.0,
            value=300.0, step=0.1,
            help="Ambient air temperature in Kelvin"
        )

        process_temp = st.slider(
            "🔥 Process Temperature (K)",
            min_value=305.0, max_value=315.0,
            value=310.0, step=0.1,
            help="Machine process temperature in Kelvin"
        )

        rpm = st.slider(
            "⚡ Rotational Speed (RPM)",
            min_value=1168, max_value=2886,
            value=1500, step=10,
            help="Spindle rotational speed"
        )

        torque = st.slider(
            "🔩 Torque (Nm)",
            min_value=3.8, max_value=76.6,
            value=40.0, step=0.5,
            help="Rotational torque applied"
        )

        tool_wear = st.slider(
            "🛠️ Tool Wear (minutes)",
            min_value=0, max_value=253,
            value=100, step=1,
            help="Cumulative tool wear time in minutes"
        )

    with col_pred:
        st.markdown("#### 🎯 Real-Time Prediction Output")

        type_encoded = le.transform([machine_type])[0]
        input_data = np.array([[type_encoded, air_temp, process_temp, rpm, torque, tool_wear]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        failure_prob = probability[1] * 100
        normal_prob = probability[0] * 100

        if failure_prob >= 70:
            risk_level = "🔴 CRITICAL RISK"
            risk_color = "#D85A30"
            action = "⚠️ IMMEDIATE SHUTDOWN — Schedule emergency maintenance now"
        elif failure_prob >= 40:
            risk_level = "🟠 HIGH RISK"
            risk_color = "#E8950A"
            action = "🔧 Schedule maintenance within 24 hours"
        elif failure_prob >= 20:
            risk_level = "🟡 MODERATE RISK"
            risk_color = "#D4A017"
            action = "📋 Monitor closely — Plan maintenance this week"
        else:
            risk_level = "🟢 NORMAL OPERATION"
            risk_color = "#1D9E75"
            action = "✅ Machine operating within safe parameters"

        st.markdown(f"""
        <div style="background-color:{risk_color}22;border-left:5px solid {risk_color};
        padding:20px;border-radius:8px;margin-bottom:15px">
        <h2 style="color:{risk_color};margin:0">{risk_level}</h2>
        <h1 style="margin:8px 0;font-size:48px">{failure_prob:.1f}%</h1>
        <p style="margin:0;font-size:14px;color:#555">Failure Probability</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Recommended Action:** {action}")
        st.markdown("---")

        fig_gauge, ax_gauge = plt.subplots(figsize=(5, 2.5))
        bar_width = 0.5
        ax_gauge.barh(['Normal', 'Failure'], [normal_prob, failure_prob],
                      color=['#1D9E75', risk_color], height=0.5, edgecolor='white')
        ax_gauge.set_xlim(0, 100)
        ax_gauge.set_xlabel("Probability (%)")
        ax_gauge.set_title("Prediction Confidence", fontweight='bold')
        for i, val in enumerate([normal_prob, failure_prob]):
            ax_gauge.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
        ax_gauge.grid(axis='x', alpha=0.3)
        st.pyplot(fig_gauge)

    st.divider()
    st.subheader("📊 Feature Impact on This Prediction")
    fig_imp, ax_imp = plt.subplots(figsize=(10, 3))
    colors_imp = ['#D85A30' if f == feature_imp.index[0] else '#185FA5' for f in feature_imp.index]
    bars_imp = ax_imp.bar(feature_imp.index, feature_imp.values, color=colors_imp, edgecolor='white')
    ax_imp.set_ylabel("Importance Score")
    ax_imp.set_title("Random Forest Feature Importance — What Drives Failure?", fontweight='bold')
    for bar, val in zip(bars_imp, feature_imp.values):
        ax_imp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9)
    ax_imp.grid(axis='y', alpha=0.3)
    st.pyplot(fig_imp)

with tab2:
    st.subheader("📊 Fleet-Wide Analytics Dashboard")

    total_machines = len(df)
    total_failures = df['Target'].sum()
    failure_rate = (total_failures / total_machines) * 100
    normal_ops = total_machines - total_failures

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Machine Records", f"{total_machines:,}")
    col2.metric("Confirmed Failures", f"{total_failures:,}")
    col3.metric("Failure Rate", f"{failure_rate:.1f}%")
    col4.metric("Normal Operations", f"{normal_ops:,}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Failure Distribution by Type")
        type_counts = df.groupby(['Type', 'Target']).size().unstack(fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(type_counts.index))
        width = 0.35
        bars_n = ax2.bar(x - width/2, type_counts[0], width, label='Normal', color='#1D9E75', edgecolor='white')
        bars_f = ax2.bar(x + width/2, type_counts[1], width, label='Failure', color='#D85A30', edgecolor='white')
        ax2.set_xlabel("Machine Type")
        ax2.set_ylabel("Count")
        ax2.set_title("Normal vs Failure by Machine Type", fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(type_counts.index)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        st.pyplot(fig2)

    with col_b:
        st.subheader("Failure Mode Breakdown")
        if 'Failure Type' in df.columns:
            failure_df = df[df['Target'] == 1]
            failure_types = failure_df['Failure Type'].value_counts()
            failure_types = failure_types[failure_types.index != 'No Failure']
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            colors3 = ['#D85A30', '#E8950A', '#185FA5', '#1D9E75', '#888780']
            wedges, texts, autotexts = ax3.pie(
                failure_types.values, labels=None,
                autopct='%1.1f%%', colors=colors3[:len(failure_types)],
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(width=0.6)
            )
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
            ax3.legend(wedges, failure_types.index,
                       title="Failure Types", loc="center left",
                       bbox_to_anchor=(1.0, 0, 0.5, 1), fontsize=8)
            ax3.set_title("Mechanical Failure Mode Distribution", fontweight='bold')
            st.pyplot(fig3)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Tool Wear vs Failure Risk")
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        normal_wear = df[df['Target'] == 0]['Tool wear [min]']
        failure_wear = df[df['Target'] == 1]['Tool wear [min]']
        ax4.hist(normal_wear, bins=40, alpha=0.6, color='#1D9E75', label='Normal', edgecolor='white')
        ax4.hist(failure_wear, bins=40, alpha=0.8, color='#D85A30', label='Failure', edgecolor='white')
        ax4.set_xlabel("Tool Wear (minutes)")
        ax4.set_ylabel("Count")
        ax4.set_title("Tool Wear Distribution — Normal vs Failure", fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        st.pyplot(fig4)

    with col_d:
        st.subheader("Temperature vs RPM Anomaly Map")
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        normal_data = df[df['Target'] == 0]
        failure_data = df[df['Target'] == 1]
        ax5.scatter(normal_data['Air temperature [K]'],
                    normal_data['Rotational speed [rpm]'],
                    alpha=0.2, s=5, color='#1D9E75', label='Normal')
        ax5.scatter(failure_data['Air temperature [K]'],
                    failure_data['Rotational speed [rpm]'],
                    alpha=0.7, s=15, color='#D85A30', label='Failure', zorder=5)
        ax5.set_xlabel("Air Temperature (K)")
        ax5.set_ylabel("Rotational Speed (RPM)")
        ax5.set_title("ML Anomaly Detection — Temp vs Speed Clusters", fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        st.pyplot(fig5)

    st.subheader("🚨 High-Risk Machine Action Checklist")
    st.caption("Machines with highest failure probability — sorted by risk score descending")

    df_scored = df.copy()
    df_scored['Type_Encoded'] = le.transform(df_scored['Type'])
    X_all = df_scored[features]
    df_scored['Failure_Probability'] = model.predict_proba(X_all)[:, 1]
    df_scored['Risk_Status'] = df_scored['Failure_Probability'].apply(
        lambda x: '🔴 CRITICAL' if x >= 0.7 else ('🟠 HIGH RISK' if x >= 0.4 else '🟡 MODERATE')
    )

    high_risk = df_scored[df_scored['Failure_Probability'] >= 0.4].sort_values(
        'Failure_Probability', ascending=False).head(20)

    display_cols = ['UDI', 'Type', 'Air temperature [K]', 'Rotational speed [rpm]',
                    'Torque [Nm]', 'Tool wear [min]', 'Failure_Probability', 'Risk_Status']
    available_cols = [c for c in display_cols if c in high_risk.columns]
    risk_display = high_risk[available_cols].copy()
    risk_display['Failure_Probability'] = risk_display['Failure_Probability'].map('{:.1%}'.format)
    st.dataframe(risk_display, use_container_width=True)

with tab3:
    st.subheader("🧠 Model Performance Metrics")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model Type", "Random Forest")
    col_m2.metric("Precision Score", f"{precision:.1%}")
    col_m3.metric("F1 Score", f"{f1:.1%}")
    col_m4.metric("Training Records", "8,000")

    st.divider()
    col_p, col_q = st.columns(2)

    with col_p:
        st.subheader("Confusion Matrix")
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        im = ax6.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax6)
        classes = ['Normal (0)', 'Failure (1)']
        tick_marks = np.arange(len(classes))
        ax6.set_xticks(tick_marks)
        ax6.set_xticklabels(classes)
        ax6.set_yticks(tick_marks)
        ax6.set_yticklabels(classes)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax6.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center", fontsize=14, fontweight='bold',
                         color="white" if cm[i, j] > thresh else "black")
        ax6.set_ylabel('Actual Label', fontweight='bold')
        ax6.set_xlabel('Predicted Label', fontweight='bold')
        ax6.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
        st.pyplot(fig6)

    with col_q:
        st.subheader("Why These Metrics Matter")
        st.markdown("""
        **Why Precision (not Accuracy)?**

        The dataset has a severe class imbalance — only **3.4% failures** out of 10,000 records.
        A naive model that always predicts "Normal" would get 96.6% accuracy but catch zero failures.

        **Our model's real value:**
        - **High Precision** = When we flag a machine, it actually needs maintenance
        - Minimises false alarms that waste engineer time on the factory floor
        - **F1 Score** balances precision vs recall for imbalanced datasets

        **Class Weights = 'balanced'** — forces the model to pay extra attention to the minority failure class, replicating real industrial ML deployment standards.

        **Business Impact:**
        - Unplanned downtime costs factories **$250,000/hour** on average
        - Early detection prevents catastrophic equipment failure
        - This model catches failures **before** they happen
        """)

    st.subheader("Feature Importance Deep Dive")
    fig7, ax7 = plt.subplots(figsize=(10, 4))
    feature_labels = ['Type', 'Air Temp (K)', 'Process Temp (K)', 'RPM', 'Torque (Nm)', 'Tool Wear (min)']
    imp_sorted = feature_imp.copy()
    colors7 = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(imp_sorted)))
    bars7 = ax7.barh(feature_labels[::-1], imp_sorted.values[::-1], color=colors7, edgecolor='white')
    ax7.set_xlabel("Importance Score")
    ax7.set_title("What Drives Machine Failure? — Feature Importance Ranking", fontweight='bold')
    for bar, val in zip(bars7, imp_sorted.values[::-1]):
        ax7.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)
    ax7.grid(axis='x', alpha=0.3)
    st.pyplot(fig7)

st.divider()
st.caption("🏭 Built by Prem Chavan | Predictive Maintenance ML Platform | github.com/premchavan772005-spec")
