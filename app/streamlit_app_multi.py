import streamlit as st
import pandas as pd
import joblib
import os
import io
import numpy as np

# Загрузка данных
history = pd.read_csv('train.csv')
df = pd.read_csv('test.csv')
unique_rows = df['row_id'].unique()
feature_columns = [
    'week_number', 'month', 'quarter', 'lag_1', 'lag_2', 'lag_3', 'lag_4',
    'lag_5', 'lag_7', 'rolling_mean_3', 'rolling_std_3', 'rolling_mean_5',
    'rolling_std_5', 'rolling_mean_7', 'rolling_std_7', 'trend_3',
    'momentum'
]

# Session state для выбранных рядов
if 'selected_rows' not in st.session_state:
    st.session_state.selected_rows = []

# UI: выбрать ряд и добавить его
row_id = st.selectbox('Выберите временной ряд', unique_rows)

if st.button('Добавить ряд'):
    if row_id not in st.session_state.selected_rows:
        st.session_state.selected_rows.append(row_id)

# Отображение выбранных рядов
st.write("Выбранные ряды:")
st.write(st.session_state.selected_rows)

# Кнопка для расчёта прогноза (только один вызов кнопки!)
pressed = st.button("Получить прогноз")

if pressed:
    if not st.session_state.selected_rows:
        st.warning("Сначала выберите хотя бы один ряд (кнопкой выше).")
    else:
        all_final = []

        for rid in st.session_state.selected_rows:
            # Прогноз по test.csv
            test_row = df[df['row_id'] == rid].copy()
            if test_row.empty:
                st.warning(f"В тестовом наборе нет ряда {rid}")
                continue

            X_test = test_row[feature_columns]
            model_path = f'models/model_{rid}.joblib'
            if not os.path.exists(model_path):
                st.warning(f'Модель для ряда {rid} не найдена!')
                continue

            model = joblib.load(model_path)
            preds = model.predict(X_test)

            result = test_row.copy()
            result['forecast'] = preds
            result = result.drop(columns=feature_columns, errors='ignore')
            result['Sum of Plan'] = np.nan
            result['fact'] = np.nan

            # История по train.csv
            history_row = history[history['row_id'] == rid].copy()
            if not history_row.empty:
                history_row = history_row.drop(columns=feature_columns, errors='ignore')
                history_row['forecast'] = np.nan
                final_df = pd.concat([history_row, result], axis=0, ignore_index=True)
            else:
                final_df = result

            all_final.append(final_df)

            with st.expander(f"Данные и прогноз для ряда: {rid}"):
                st.dataframe(final_df)

        # Сохраняем в Excel (по листу на каждый ряд)
        if all_final:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for i, rid in enumerate(st.session_state.selected_rows):
                    if i < len(all_final):
                        sheet_name = f'{str(rid)[:20]}'
                        all_final[i].to_excel(writer, index=False, sheet_name=sheet_name)
            st.download_button(
                "Скачать все прогнозы (Excel)",
                data=output.getvalue(),
                file_name='all_forecasts.xlsx'
            )
        else:
            st.info("Нет данных для выбранных рядов.")

# Кнопка для очистки списка рядов
if st.button("Очистить список рядов"):
    st.session_state.selected_rows = []
