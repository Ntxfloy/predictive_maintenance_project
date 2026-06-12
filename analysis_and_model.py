import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

# Page Title
st.title("📊 Анализ данных и моделирование отказов оборудования")
st.markdown("""
Этот дашборд разработан для предиктивного обслуживания оборудования с использованием набора данных **AI4I 2020 Predictive Maintenance**.
Он позволяет исследовать характеристики датчиков, обучать классификаторы и прогнозировать риски аварий в режиме реального времени.
""")

# Load Dataset
@st.cache_data(show_spinner="Загрузка данных из репозитория UCI...")
def load_data():
    try:
        dataset = fetch_ucirepo(id=601)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()
        ids = dataset.data.ids.copy()
        
        # Combine into single DataFrame for analysis
        df_full = pd.concat([ids, features, targets], axis=1)
        return df_full, features, targets
    except Exception as e:
        st.error(f"Не удалось загрузить данные из UCI: {e}")
        return None, None, None

df_full, features, targets = load_data()

if df_full is not None:
    # 3. Mapping and Basic Cleanup
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df_clean = df_full.copy()
    df_clean['Type_Code'] = df_clean['Type'].map(type_mapping)
    
    # Define features and target for models
    feature_cols = ['Type_Code', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    target_col = 'Machine failure'
    
    # Create Tabs
    tab_eda, tab_train, tab_predict = st.tabs(["🔍 Анализ данных (EDA)", "🤖 Обучение моделей", "⚙️ Прогнозирование в реальном времени"])
    
    with tab_eda:
        st.header("Анализ исходного набора данных")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего наблюдений", len(df_full))
        col2.metric("Количество отказов", int(df_full['Machine failure'].sum()))
        col3.metric("Процент аварий", f"{(df_full['Machine failure'].mean() * 100):.2f}%")
        
        st.subheader("Просмотр первых строк датасета")
        st.dataframe(df_full.head(10))
        
        st.subheader("Статистическое описание данных")
        st.dataframe(df_full.describe())
        
        st.subheader("Распределение отказов по типам")
        failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        sum_failures = df_full[failure_types].sum().reset_index()
        sum_failures.columns = ['Тип отказа', 'Количество случаев']
        
        fig_fail, ax_fail = plt.subplots(figsize=(8, 4))
        sns.barplot(data=sum_failures, x='Тип отказа', y='Количество случаев', palette='Oranges_r', ax=ax_fail)
        ax_fail.set_title("Частота различных видов отказов оборудования")
        for p in ax_fail.patches:
            ax_fail.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        st.pyplot(fig_fail)
        plt.close()

        # Correlation Heatmap
        st.subheader("Корреляционная матрица числовых признаков")
        numeric_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Machine failure']
        corr_matrix = df_clean[numeric_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
        st.pyplot(fig_corr)
        plt.close()

        # Distribution plot for sensors
        st.subheader("Распределения сенсорных показаний по классам")
        sensor_to_plot = st.selectbox("Выберите датчик для анализа:", ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear'])
        fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
        sns.kdeplot(data=df_clean, x=sensor_to_plot, hue='Machine failure', fill=True, common_norm=False, palette='Set1', alpha=0.5, ax=ax_dist)
        ax_dist.set_title(f"Плотность распределения: {sensor_to_plot}")
        st.pyplot(fig_dist)
        plt.close()
        
    with tab_train:
        st.header("Обучение классификаторов")
        st.markdown("Мы обучаем три модели: **Логистическую регрессию**, **Случайный лес** и **XGBoost**.")
        
        # Prepare Data
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        # Fit scaler on train, scale train & test (except Type_Code)
        scale_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])
        
        # Let's cache the trained models to avoid re-running on every tab switch
        @st.cache_resource
        def train_models(X_tr, y_tr):
            # 1. Logistic Regression
            lr = LogisticRegression(class_weight='balanced', random_state=42)
            lr.fit(X_tr, y_tr)
            
            # 2. Random Forest
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)
            rf.fit(X_tr, y_tr)
            
            # 3. XGBoost
            scale_pos = (len(y_tr) - sum(y_tr)) / sum(y_tr)
            xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos, learning_rate=0.1, max_depth=5, eval_metric='logloss', random_state=42)
            xgb.fit(X_tr, y_tr)
            
            return lr, rf, xgb

        lr_model, rf_model, xgb_model = train_models(X_train_scaled, y_train)
        
        # Evaluations
        models = {
            "Logistic Regression": lr_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model
        }
        
        eval_results = []
        
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            eval_results.append({
                "Модель": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc
            })
            
        df_results = pd.DataFrame(eval_results)
        st.subheader("Сводная таблица метрик на тестовой выборке")
        st.dataframe(df_results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }))
        
        # Feature Importance for XGBoost
        st.subheader("Важность признаков (XGBoost)")
        importances = xgb_model.feature_importances_
        df_imp = pd.DataFrame({
            "Признак": ['Тип продукта', 'Температура воздуха', 'Рабочая температура', 'Скорость вращения', 'Крутящий момент', 'Износ инструмента'],
            "Важность": importances
        }).sort_values(by="Важность", ascending=False)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_imp, x='Важность', y='Признак', palette='viridis', ax=ax_imp)
        ax_imp.set_title("Относительное влияние признаков на прогноз отказа")
        st.pyplot(fig_imp)
        plt.close()

        # ROC Curve Comparison
        st.subheader("ROC-кривые моделей")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
        for name, model in models.items():
            probs = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', label="Случайный выбор")
        ax_roc.set_xlabel("FPR (Доля ложных тревог)")
        ax_roc.set_ylabel("TPR (Чувствительность)")
        ax_roc.set_title("Сравнение ROC-кривых на тестовой выборке")
        ax_roc.legend()
        st.pyplot(fig_roc)
        plt.close()

        # Confusion Matrix for Selected Model
        st.subheader("Матрица ошибок")
        sel_model_name = st.selectbox("Выберите модель для детального анализа ошибок:", list(models.keys()))
        sel_model = models[sel_model_name]
        y_pred_sel = sel_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred_sel)
        
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Исправно', 'Отказ'], yticklabels=['Исправно', 'Отказ'], ax=ax_cm)
        ax_cm.set_xlabel("Предсказано моделью")
        ax_cm.set_ylabel("Реальное состояние")
        ax_cm.set_title(f"Матрица ошибок: {sel_model_name}")
        st.pyplot(fig_cm)
        plt.close()
        
    with tab_predict:
        st.header("Симуляция работы оборудования")
        st.markdown("Задайте текущие показатели датчиков с панели слева или снизу, чтобы рассчитать вероятность поломки.")
        
        col_inp, col_out = st.columns([1, 1])
        
        with col_inp:
            st.subheader("Текущая телеметрия")
            # User inputs
            val_type = st.selectbox("Тип качества детали (Type)", ["L (Low)", "M (Medium)", "H (High)"], index=0)
            val_air = st.slider("Температура воздуха [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
            val_process = st.slider("Рабочая температура процесса [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
            val_speed = st.slider("Скорость вращения [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
            val_torque = st.slider("Крутящий момент [Nm]", min_value=0.0, max_value=80.0, value=40.0, step=0.5)
            val_wear = st.slider("Износ инструмента [min]", min_value=0, max_value=250, value=120, step=1)
            
            # Map quality string to mapping code
            type_code = type_mapping[val_type[0]]
            
        with col_out:
            st.subheader("Прогноз состояния")
            
            # Prepare single row for prediction
            input_df = pd.DataFrame([{
                'Type_Code': type_code,
                'Air temperature': val_air,
                'Process temperature': val_process,
                'Rotational speed': val_speed,
                'Torque': val_torque,
                'Tool wear': val_wear
            }])
            
            # Scale numeric inputs using the same fitted scaler
            input_scaled = input_df.copy()
            input_scaled[scale_cols] = scaler.transform(input_df[scale_cols])
            
            # Perform prediction using XGBoost (highest performance)
            pred_class = xgb_model.predict(input_scaled)[0]
            pred_prob = xgb_model.predict_proba(input_scaled)[0][1]
            
            st.metric("Вероятность аварийного отказа", f"{pred_prob * 100:.2f}%")
            
            if pred_class == 1 or pred_prob > 0.5:
                st.error("🚨 ВНИМАНИЕ: Высокий риск отказа оборудования!")
                st.markdown("""
                **Рекомендуемые действия:**
                1. Срочно остановить станок для плановой проверки.
                2. Проверить износ режущего инструмента.
                3. Проконтролировать систему охлаждения (разница температур воздуха и процесса критична).
                """)
            else:
                st.success("✅ Состояние стабильное. Оборудование работает в штатном режиме.")
                st.markdown("""
                Показатели телеметрии находятся в пределах допустимых физических норм.
                Продолжайте плановый мониторинг.
                """)
                
            # Physics validation details
            temp_diff = val_process - val_air
            power = (2 * np.pi * val_speed / 60) * val_torque
            
            st.info(f"""
            **Физические вычисления:**
            - **Разность температур**: {temp_diff:.1f} K
            - **Потребляемая мощность**: {power:.1f} Вт
            """)
