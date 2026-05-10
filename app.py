import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import json
import pickle
import os
from datetime import datetime
import hashlib
import io

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_cyrillic_fonts():
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('DejaVuSans', path))
                return 'DejaVuSans'
            except:
                continue
    return None


CYRILLIC_FONT = register_cyrillic_fonts() or 'Helvetica'


class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_placeholder):
        super().__init__()
        self.plot_placeholder = plot_placeholder
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss', 0))
        self.accs.append(logs.get('accuracy', 0))
        if epoch % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.plot(self.losses, label='Loss', color='red')
            ax1.set_title('Loss')
            ax1.grid(True)
            ax2.plot(self.accs, label='Accuracy', color='green')
            ax2.set_title('Accuracy')
            ax2.grid(True)
            self.plot_placeholder.pyplot(fig)
            plt.close(fig)


class BankruptcyPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = 'model.keras'
        self.scaler_path = 'scaler.pkl'
        self.load_resources()

    def load_resources(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

    def get_industry_benchmarks(self):
        return {
            "Строительство": {
                "wc_ta": 0.15, "ebit_ta": 0.09, "debt_ta": 0.65, "sales_ta": 1.8,
                "current_ratio": 1.5, "roe": 0.12, "profit_margin": 0.08
            },
            "Торговля": {
                "wc_ta": 0.25, "ebit_ta": 0.12, "debt_ta": 0.55, "sales_ta": 3.2,
                "current_ratio": 1.8, "roe": 0.18, "profit_margin": 0.06
            },
            "Производство": {
                "wc_ta": 0.18, "ebit_ta": 0.11, "debt_ta": 0.60, "sales_ta": 1.5,
                "current_ratio": 1.6, "roe": 0.15, "profit_margin": 0.10
            },
            "IT и услуги": {
                "wc_ta": 0.35, "ebit_ta": 0.22, "debt_ta": 0.40, "sales_ta": 2.1,
                "current_ratio": 2.2, "roe": 0.25, "profit_margin": 0.15
            },
            "Сельское хозяйство": {
                "wc_ta": 0.12, "ebit_ta": 0.07, "debt_ta": 0.70, "sales_ta": 0.9,
                "current_ratio": 1.3, "roe": 0.08, "profit_margin": 0.05
            },
            "Транспорт": {
                "wc_ta": 0.10, "ebit_ta": 0.08, "debt_ta": 0.68, "sales_ta": 1.2,
                "current_ratio": 1.4, "roe": 0.10, "profit_margin": 0.07
            },
            "Другая": {
                "wc_ta": 0.20, "ebit_ta": 0.10, "debt_ta": 0.60, "sales_ta": 1.5,
                "current_ratio": 1.5, "roe": 0.12, "profit_margin": 0.08
            }
        }

    def compare_with_industry(self, company_data: dict, industry: str):
        benchmarks = self.get_industry_benchmarks()
        bm = benchmarks.get(industry, benchmarks["Производство"])
        ta = company_data.get('ta', 1) or 1

        return {
            "industry": industry,
            "metrics": {
                "Оборотный капитал / Активы": {
                    "company": round(company_data.get('wc', 0) / ta, 3),
                    "benchmark": bm["wc_ta"],
                    "better": company_data.get('wc', 0) / ta > bm["wc_ta"],
                    "difference": round(company_data.get('wc', 0) / ta - bm["wc_ta"], 3)
                },
                "EBIT / Активы": {
                    "company": round(company_data.get('ebit', 0) / ta, 3),
                    "benchmark": bm["ebit_ta"],
                    "better": company_data.get('ebit', 0) / ta > bm["ebit_ta"],
                    "difference": round(company_data.get('ebit', 0) / ta - bm["ebit_ta"], 3)
                },
                "Долг / Активы": {
                    "company": round(company_data.get('tl', 0) / ta, 3),
                    "benchmark": bm["debt_ta"],
                    "better": company_data.get('tl', 0) / ta < bm["debt_ta"],
                    "difference": round(company_data.get('tl', 0) / ta - bm["debt_ta"], 3)
                },
                "Выручка / Активы": {
                    "company": round(company_data.get('sales', 0) / ta, 3),
                    "benchmark": bm["sales_ta"],
                    "better": company_data.get('sales', 0) / ta > bm["sales_ta"],
                    "difference": round(company_data.get('sales', 0) / ta - bm["sales_ta"], 3)
                },
                "Текущая ликвидность": {
                    "company": round(company_data.get('ca', 0) / max(company_data.get('cl', 1), 1), 3),
                    "benchmark": bm["current_ratio"],
                    "better": company_data.get('ca', 0) / max(company_data.get('cl', 1), 1) > bm["current_ratio"],
                    "difference": round(
                        company_data.get('ca', 0) / max(company_data.get('cl', 1), 1) - bm["current_ratio"], 3)
                }
            }
        }

    def create_radar_chart(self, data: dict, company_name: str = "Компания"):
        ratios = {
            "WC/TA": data.get('wc', 0) / (data.get('ta', 1) or 1),
            "EBIT/TA": data.get('ebit', 0) / (data.get('ta', 1) or 1),
            "Sales/TA": data.get('sales', 0) / (data.get('ta', 1) or 1),
            "TL/TA": data.get('tl', 0) / (data.get('ta', 1) or 1),
            "EBIT/Sales": data.get('ebit', 0) / (data.get('sales', 1) or 1),
            "Equity/TA": data.get('equity', 0) / (data.get('ta', 1) or 1),
        }

        values = [
            min(1.8, max(0, ratios["WC/TA"] * 2.5)),
            min(1.8, max(0, ratios["EBIT/TA"] * 10)),
            min(1.8, max(0, ratios["Sales/TA"] / 1.5)),
            min(1.8, max(0, 1.8 - ratios["TL/TA"] * 2)),
            min(1.8, max(0, ratios["EBIT/Sales"] * 6)),
            min(1.8, max(0, ratios["Equity/TA"] * 2.5)),
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values, theta=list(ratios.keys()), fill='toself',
            name=company_name, line_color='#1E88E5'
        ))
        fig.update_layout(
            title=f"Финансовый профиль - {company_name}",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.8])),
            height=500, showlegend=False
        )
        return fig

    def calculate_feature_importance(self, features17, model_choice, current_data=None):
        if current_data is None:
            current_data = {}

        feature_names = [
            "WC", "TA", "RE", "EBIT", "MV", "BV", "Sales", "TL",
            "PBT", "CL", "CA", "Loss", "Equity", "Cred_Debt",
            "Deb_Debt", "Loan", "Sales_Profit"
        ]

        if model_choice == "AI (TensorFlow Neural Network)" and self.model is not None:
            try:
                features_array = np.array(features17, dtype=np.float32).reshape(1, -1)

                try:
                    base_scaled = self.scaler.transform(features_array)
                except:
                    base_scaled = features_array

                base_prob = float(self.model.predict(base_scaled, verbose=0)[0][0])

                importances = []
                n_tests = 80

                for i in range(len(feature_names)):
                    perturbed = np.tile(base_scaled, (n_tests, 1))
                    noise = np.random.normal(0, 0.15, n_tests) * max(abs(base_scaled[0, i]), 1e-5)
                    perturbed[:, i] = base_scaled[0, i] + noise

                    perturbed_preds = self.model.predict(perturbed, verbose=0).flatten()
                    prob_changes = np.abs(perturbed_preds - base_prob)
                    importance = float(np.mean(prob_changes))
                    importances.append(importance)

                importances = np.array(importances)
                if importances.sum() > 0:
                    importances = importances / importances.sum()
                else:
                    importances = np.ones(len(feature_names)) / len(feature_names)

                imp_dict = dict(zip(feature_names, np.round(importances, 4)))
                return imp_dict

            except Exception as e:
                st.warning(f"AI важность признаков не сработала: {e}")
                return self._get_simple_importance(feature_names)

        return self._get_classical_importance(feature_names, model_choice, current_data)

    def _get_simple_importance(self, feature_names):
        n = len(feature_names)
        return dict(zip(feature_names, [round(1 / n, 4)] * n))

    def _get_classical_importance(self, feature_names, model_choice, current_data):
        ta = current_data.get('ta', 1.0)
        tl_ratio = current_data.get('tl', 0) / ta if ta > 0 else 0
        wc_ratio = current_data.get('wc', 0) / ta if ta > 0 else 0
        ebit_ratio = current_data.get('ebit', 0) / ta if ta > 0 else 0

        if model_choice == "Альтмана":
            imp = {
                "EBIT/TA": 0.32 + ebit_ratio * 0.25,
                "WC/TA": 0.24 + wc_ratio * 0.35,
                "RE/TA": 0.18,
                "TL/TA": 0.13 + tl_ratio * 0.30,
                "Sales/TA": 0.13
            }
        elif model_choice == "Спрингейта":
            imp = {
                "EBIT/TA": 0.38 + ebit_ratio * 0.35,
                "WC/TA": 0.25 + wc_ratio * 0.30,
                "EBIT/TL": 0.20,
                "Sales/TA": 0.17
            }
        elif model_choice == "Таффлера":
            imp = {
                "EBIT/CL": 0.30,
                "CA/TL": 0.23,
                "CL/TA": 0.20 + tl_ratio * 0.25,
                "Sales/TA": 0.17,
                "WC/TA": 0.10
            }
        elif model_choice == "Зайцевой":
            imp = {
                "Кредиторская задолженность": 0.26 + tl_ratio * 0.35,
                "Убыток/Капитал": 0.22,
                "Выручка/Активы": 0.18,
                "Заёмные средства": 0.20,
                "EBIT/TA": 0.14
            }
        else:
            imp = {"EBIT/TA": 0.30, "WC/TA": 0.25, "TL/TA": 0.20,
                   "Sales/TA": 0.15, "RE/TA": 0.10}

        total = sum(imp.values())
        imp = {k: round(v / total, 4) for k, v in imp.items()}
        return imp

    def create_feature_importance_chart(self, importance_dict: dict, model_choice: str):
        df = pd.DataFrame({
            "Признак": list(importance_dict.keys()),
            "Важность": list(importance_dict.values())
        })
        df = df.sort_values("Важность", ascending=True)

        fig = px.bar(df, x="Важность", y="Признак", orientation='h',
                     title=f"Важность факторов - {model_choice}",
                     color="Важность",
                     color_continuous_scale='Blues_r',
                     text_auto='.3f')

        fig.update_layout(height=460, xaxis_title="Относительная важность (%)")
        fig.update_traces(textposition='outside')
        return fig

    def altman_z_score(self, wc, ta, re, ebit, mv, bv, sales):
        if ta == 0 or bv == 0:
            return {"score": 0, "risk": "Ошибка данных", "probability": 0.0}
        x1 = wc / ta
        x2 = re / ta
        x3 = ebit / ta
        x4 = mv / bv
        x5 = sales / ta
        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        if z > 2.99:
            prob = max(0.03, 0.25 - (z - 2.99) * 0.08)
            risk = "Низкий риск"
        elif z > 1.81:
            prob = 0.25 + (2.99 - z) * 0.25
            risk = "Серая зона"
        else:
            prob = 0.55 + min((1.81 - z) * 0.25, 0.40)
            risk = "Высокий риск"
        return {"score": round(z, 4), "risk": risk, "probability": round(prob, 4)}

    def springate_model(self, wc, ta, ebit, tl, sales):
        if ta == 0 or tl == 0:
            return {"score": 0, "risk": "Ошибка данных", "probability": 0.0}
        x1 = wc / ta
        x2 = ebit / ta
        x3 = ebit / tl
        x4 = sales / ta
        z = 1.03 * x1 + 3.07 * x2 + 0.66 * x3 + 0.4 * x4
        if z > 0.862:
            prob = max(0.05, 0.30 - (z - 0.862) * 0.25)
            risk = "Низкий риск"
        else:
            prob = 0.30 + min((0.862 - z) * 0.80, 0.65)
            risk = "Высокий риск"
        return {"score": round(z, 4), "risk": risk, "probability": round(prob, 4)}

    def taffler_model(self, pbt, cl, ca, tl, sales, ta):
        if cl == 0 or ta == 0 or tl == 0:
            return {"score": 0, "risk": "Ошибка данных", "probability": 0.0}
        x1 = pbt / cl
        x2 = ca / tl
        x3 = cl / ta
        x4 = sales / ta
        z = 0.53 * x1 + 0.13 * x2 + 0.18 * x3 + 0.16 * x4
        if z > 0.3:
            prob = max(0.08, 0.25 - (z - 0.3) * 0.40)
            risk = "Низкий риск"
        elif z > 0.2:
            prob = 0.25 + (0.3 - z) * 1.5
            risk = "Серая зона"
        else:
            prob = 0.40 + min((0.2 - z) * 1.2, 0.55)
            risk = "Высокий риск"
        return {"score": round(z, 4), "risk": risk, "probability": round(prob, 4)}

    def zaitseva_model(self, loss, equity, cred_debt, deb_debt, revenue, assets, loan, sales_profit):

        if equity <= 0 or assets <= 0 or revenue <= 0:
            return {"score": 0, "risk": "Недостаточно данных (отриц. капитал или выручка)", "probability": 0.0}

        x1 = max(0, loss) / equity

        x2 = cred_debt / max(deb_debt, 1)

        x3 = (cred_debt + loan) / assets

        x4 = max(0, loss) / revenue

        x5 = (cred_debt + loan) / equity

        x6 = assets / revenue

        k = 0.25 * x1 + 0.1 * x2 + 0.2 * x3 + 0.25 * x4 + 0.1 * x5 + 0.1 * x6

        kn = 1.6

        if k > kn:
            prob = 0.7 + min((k - kn) * 0.1, 0.25)
            risk = "Высокий риск (K > Kn)"
        elif k > kn * 0.8:
            prob = 0.4 + (k - kn * 0.8) * 0.5
            risk = "Средний риск / Серая зона"
        else:
            prob = 0.1 + (k / kn) * 0.2
            risk = "Низкий риск"

        return {"score": round(k, 4), "risk": risk, "probability": round(prob, 4)}

    def build_neural_network(self, input_dim=17):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(48, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(24, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),

            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_model(self, data_path, progress_placeholder):
        try:
            df = pd.read_csv(data_path, na_values='?', sep=',', header=None, skiprows=1)

            if df.shape[1] != 18:
                st.error(f"Неверное количество колонок: {df.shape[1]}. Ожидается 18.")
                return False

            df.fillna(0, inplace=True)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.astype(int)

            X_scaled = self.scaler.fit_transform(X)

            self.model = self.build_neural_network()

            self.model.fit(
                X_scaled, y,
                epochs=100,
                batch_size=16,
                verbose=0,
                callbacks=[StreamlitCallback(progress_placeholder)],
                validation_split=0.2
            )

            self.model.save(self.model_path)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            return True
        except Exception as e:
            st.error(f"Ошибка обучения: {str(e)}")
            return False



    def predict_bankruptcy_ml(self, features17):
        if self.model is None:
            raise ValueError("Модель ещё не обучена! Перейдите во вкладку 'Обучение модели'.")

        features_array = np.array(features17, dtype=np.float32).reshape(1, -1)

        try:
            features_scaled = self.scaler.transform(features_array)
        except:
            st.warning("⚠️ Скалер не обучен.")
            features_scaled = features_array

        raw_prob = float(self.model.predict(features_scaled, verbose=0)[0][0])

        calibrated = raw_prob ** 1.6
        calibrated = 0.5 * calibrated + 0.5 * raw_prob

        final_prob = 1 / (1 + np.exp(-5 * (calibrated - 0.5)))
        final_prob = float(np.clip(final_prob, 0.05, 0.85))

        if final_prob > 0.65:
            risk = "Высокий риск"
        elif final_prob > 0.40:
            risk = "Средний риск"
        else:
            risk = "Низкий риск"

        return {
            "score": round(final_prob, 4),
            "risk": risk,
            "probability": final_prob
        }

    def forecast_bankruptcy_improved(self, current_data: dict, years: int = 3,
                                     scenario: str = "base"):
        scenarios = {
            "Оптимистичный": {"sales_growth": 0.12, "ebit_margin": 0.145, "ta_growth": 0.08,
                           "debt_growth": 0.05, "wc_days": 45, "interest_rate": 0.09},
            "Базовый": {"sales_growth": 0.08, "ebit_margin": 0.115, "ta_growth": 0.06,
                     "debt_growth": 0.07, "wc_days": 60, "interest_rate": 0.105},
            "Пессимистичный": {"sales_growth": 0.02, "ebit_margin": 0.06, "ta_growth": 0.03,
                            "debt_growth": 0.12, "wc_days": 85, "interest_rate": 0.14}
        }
        scenario_key = scenario.strip().capitalize()
        params = scenarios.get(scenario, scenarios["Базовый"])
        forecast_results = []
        current = current_data.copy()

        current_sales = current.get('sales', 1000000)
        current_ta = current.get('ta', 1000000)
        current_tl = current.get('tl', current_ta * 0.5)
        current_re = current.get('re', current_ta * 0.3)

        for year in range(1, years + 1):
            projected = {}
            projected['sales'] = current_sales * (1 + params['sales_growth']) ** year
            projected['ebit'] = projected['sales'] * params['ebit_margin']
            projected['ta'] = current_ta * (1 + params['ta_growth']) ** year
            projected['tl'] = current_tl * (1 + params['debt_growth']) ** year
            projected['wc'] = projected['sales'] * (params['wc_days'] / 365)

            avg_debt = (current_tl + projected['tl']) / 2
            interest_expense = avg_debt * params['interest_rate']

            pbt = projected['ebit'] - interest_expense
            tax = max(0, pbt * 0.20)
            net_profit = pbt - tax

            projected['re'] = current_re + net_profit * 0.70
            projected['equity'] = projected['ta'] - projected['tl']

            wc_ta = projected['wc'] / projected['ta'] if projected['ta'] > 0 else 0
            tl_ta = projected['tl'] / projected['ta'] if projected['ta'] > 0 else 0
            ebit_ta = projected['ebit'] / projected['ta'] if projected['ta'] > 0 else 0

            risk_score = 0.0
            risk_score += max(0, -wc_ta) * 0.35
            risk_score += max(0, 0.15 - wc_ta) * 0.22
            risk_score += max(0, -ebit_ta) * 0.45
            risk_score += max(0, 0.04 - ebit_ta) * 0.28
            risk_score += max(0, tl_ta - 0.60) * 0.40
            risk_score += max(0, tl_ta - 0.80) * 0.35

            risk_score = np.clip(risk_score, 0, 4.5)
            probability = 1 / (1 + np.exp(2.8 - risk_score * 1.35))
            probability = float(np.clip(probability, 0.03, 0.97))

            risk = "Низкий" if probability < 0.2 else "Средний" if probability < 0.5 else "Высокий"

            forecast_results.append({
                'Год': f'+{year}',
                'Сценарий': scenario.capitalize(),
                'Выручка (млн)': round(projected['sales'] / 1_000_000, 1),
                'EBIT (млн)': round(projected['ebit'] / 1_000_000, 1),
                'Активы (млн)': round(projected['ta'] / 1_000_000, 1),
                'Долг (млн)': round(projected['tl'] / 1_000_000, 1),
                'Чистая прибыль (млн)': round(net_profit / 1_000_000, 1),
                'WC/TA': round(wc_ta, 3),
                'TL/TA': round(tl_ta, 3),
                'EBIT/TA': round(ebit_ta, 3),
                'Вероятность': f"{probability:.1%}",
                'Риск': risk
            })

        return pd.DataFrame(forecast_results)

    def generate_recommendations(self, res: dict, model_choice: str,
                                 current_data: dict = None,
                                 importance_dict: dict = None):
        risk = res.get('risk', '')
        prob = res.get('probability', 0.0)

        rec = {
            "urgency": "",
            "main_problem": "",
            "top_factor": "",
            "actions": [],
            "plan_90_days": []
        }

        if not importance_dict or not current_data:
            rec[
                "urgency"] = "🔴 Критическая ситуация" if prob > 0.6 else "🟠 Повышенный риск" if prob > 0.3 else "🟢 Низкий риск"
            rec["actions"] = ["Провести детальный финансовый анализ", "Проверять показатели ежемесячно"]
            return rec

        top_factor = max(importance_dict, key=importance_dict.get)
        rec["top_factor"] = top_factor

        ta = current_data.get('ta', 1.0)
        value = None

        ratios = {
            "WC/TA": current_data.get('wc', 0) / ta if ta > 0 else 0,
            "EBIT/TA": current_data.get('ebit', 0) / ta if ta > 0 else 0,
            "Sales/TA": current_data.get('sales', 0) / ta if ta > 0 else 0,
            "TL/TA": current_data.get('tl', 0) / ta if ta > 0 else 0,
            "Equity/TA": current_data.get('equity', 0) / ta if ta > 0 else 0,
            "EBIT/Sales": current_data.get('ebit', 0) / max(current_data.get('sales', 1), 1),
            "CL/TA": current_data.get('cl', 0) / ta if ta > 0 else 0,
            "CA/TL": current_data.get('ca', 0) / max(current_data.get('tl', 1), 1),
            "Loss/Equity": abs(current_data.get('loss', 0)) / max(current_data.get('equity', 1), 1),
            "Cred_Debt": current_data.get('cred_debt', 0) / ta if ta > 0 else 0,
            "Loan/Equity": current_data.get('loan', 0) / max(current_data.get('equity', 1), 1),
        }

        if top_factor in ratios:
            value = ratios[top_factor]
        elif top_factor == "Loss":
            value = abs(current_data.get('loss', 0))
        elif top_factor == "Equity":
            value = current_data.get('equity', 0) / ta if ta > 0 else 0
        else:
            value = 0.5

        if value is None:
            value = 0.5

        if top_factor in ["WC/TA", "WC"]:
            if value < 0:
                rec["main_problem"] = "Отрицательный оборотный капитал"
                rec["urgency"] = "🔴 Критическая ликвидность"
                rec["actions"] = ["Срочное взыскание дебиторной задолженности", "Переговоры об отсрочках с поставщиками",
                                  "Продажа запасов"]
            elif value < 0.08:
                rec["main_problem"] = "Низкий оборотный капитал"
                rec["urgency"] = "🟠 Проблемы с ликвидностью"
                rec["actions"] = ["Оптимизация дебиторской и кредиторской задолженности", "Факторинг",
                                  "Снижение запасов"]
            else:
                rec["main_problem"] = "Оборотный капитал в норме"
                rec["actions"] = ["Поддерживать текущий уровень", "Оптимизировать оборачиваемость"]

        elif top_factor in ["EBIT/TA", "EBIT/Sales", "EBIT"]:
            if value < 0:
                rec["main_problem"] = "Операционная убыточность"
                rec["urgency"] = "🔴 Операционные убытки"
                rec["actions"] = ["Сокращение затрат на 20-30%", "Пересмотр цен", "Закрытие убыточных направлений"]
            elif value < 0.04:
                rec["main_problem"] = "Низкая операционная рентабельность"
                rec["urgency"] = "🟠 Низкая рентабельность"
                rec["actions"] = ["Оптимизация издержек", "Увеличение маржинальности", "Анализ ценовой политики"]
            else:
                rec["main_problem"] = "Рентабельность на приемлемом уровне"
                rec["actions"] = ["Поддерживать рентабельность", "Инвестировать в развитие"]

        elif top_factor in ["TL/TA", "Кредиторская задолженность", "Cred_Debt"]:
            if value > 0.75:
                rec["main_problem"] = "Критическая долговая нагрузка"
                rec["urgency"] = "🔴 Высокий долговой риск"
                rec["actions"] = ["Рефинансирование", "Реструктуризация долгов", "Поиск инвесторов"]
            elif value > 0.55:
                rec["main_problem"] = "Повышенная долговая нагрузка"
                rec["urgency"] = "🟠 Высокая долговая нагрузка"
                rec["actions"] = ["Снижение долговой нагрузки", "Увеличение собственного капитала"]
            else:
                rec["main_problem"] = "Долговая нагрузка в норме"
                rec["actions"] = ["Поддерживать текущий уровень долга"]

        elif top_factor in ["Sales/TA", "Выручка/Активы"]:
            if value < 0.7:
                rec["main_problem"] = "Низкая оборачиваемость активов"
                rec["urgency"] = "🟠 Низкая эффективность активов"
                rec["actions"] = ["Увеличение выручки", "Списание/продажа неработающих активов"]
            else:
                rec["main_problem"] = "Эффективное использование активов"
                rec["actions"] = ["Поддерживать оборачиваемость"]

        elif top_factor in ["Equity/TA", "Equity"]:
            if value < 0.2:
                rec["main_problem"] = "Низкая капитализация / высокая зависимость от заёмных средств"
                rec["urgency"] = "🟠 Низкий собственный капитал"
                rec["actions"] = ["Увеличение уставного капитала", "Нераспределённая прибыль в оборот"]
            else:
                rec["main_problem"] = "Достаточный уровень капитализации"
                rec["actions"] = ["Поддерживать структуру капитала"]

        elif top_factor in ["Loss", "Loss/Equity", "Убыток/Капитал"]:
            if value > 0:
                rec["main_problem"] = "Наличие убытков"
                rec["urgency"] = "🔴 Убыточная деятельность"
                rec["actions"] = ["Анализ причин убытков", "Снижение себестоимости", "Поиск новых рынков сбыта"]
            else:
                rec["main_problem"] = "Прибыльная деятельность"
                rec["actions"] = ["Поддерживать прибыльность"]

        elif top_factor in ["CL/TA", "CA/TL"]:
            if top_factor == "CL/TA" and value > 0.5:
                rec["main_problem"] = "Высокая доля текущих обязательств"
                rec["urgency"] = "🟠 Высокие текущие обязательства"
                rec["actions"] = ["Реструктуризация краткосрочных долгов", "Увеличение ликвидности"]
            elif top_factor == "CA/TL" and value < 1.0:
                rec["main_problem"] = "Недостаток текущих активов для покрытия обязательств"
                rec["urgency"] = "🟠 Низкая ликвидность"
                rec["actions"] = ["Увеличение оборотных активов", "Снижение текущих обязательств"]
            else:
                rec["main_problem"] = "Ликвидность на приемлемом уровне"
                rec["actions"] = ["Поддерживать ликвидность"]

        else:
            rec["main_problem"] = f"Ключевой фактор: {top_factor}"
            if prob > 0.5:
                rec["urgency"] = "🔴 Высокий риск"
                rec["actions"] = ["Детальный анализ по этому фактору", "Разработка антикризисных мер"]
            elif prob > 0.3:
                rec["urgency"] = "🟠 Повышенный риск"
                rec["actions"] = ["Проверка фактора", "Превентивные меры"]
            else:
                rec["urgency"] = "🟢 Низкий риск"
                rec["actions"] = ["Регулярная проверка"]

        if prob > 0.65 or "Критическая" in rec.get("urgency", ""):
            rec["urgency"] = "🔴 Критическая ситуация - немедленные действия"
            rec["plan_90_days"] = [
                "Неделя 1–2: Полный аудит и заморозка некритических расходов",
                "Неделя 3–4: Переговоры с кредиторами",
                "Месяц 1–2: Сокращение расходов ≥25%",
                "Месяц 2–3: Рефинансирование и взыскание дебиторной задолженности"
            ]
        elif prob > 0.3:
            rec["urgency"] = "🟠 Повышенный риск - план на 3 месяца"
            rec["plan_90_days"] = ["Еженедельная проверка ликвидности", "Работа по главному фактору", "Стресс-тестирование"]
        else:
            rec["urgency"] = "🟢 Низкий риск - профилактика"
            rec["plan_90_days"] = ["Ежемесячная проверка", "Плановый анализ", "Поддержание показателей"]

        if len(rec["actions"]) < 3:
            while len(rec["actions"]) < 3:
                rec["actions"].append("Мониторинг финансовых показателей")

        return rec


def generate_pdf_report(company_name, inn, model_choice, res, current_data, forecast_df, rec, radar_fig, imp_fig,
                        industry_comparison):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    font_name = CYRILLIC_FONT

    title_style = ParagraphStyle('Title',
                                 parent=styles['Title'],
                                 fontName=font_name,
                                 fontSize=18,
                                 spaceAfter=20,
                                 alignment=1,
                                 textColor=colors.darkblue)

    heading_style = ParagraphStyle('Heading',
                                   parent=styles['Heading2'],
                                   fontName=font_name,
                                   fontSize=14,
                                   spaceAfter=12,
                                   textColor=colors.black)

    normal_style = ParagraphStyle('Normal',
                                  parent=styles['Normal'],
                                  fontName=font_name,
                                  fontSize=11,
                                  leading=14)

    story = []

    story.append(Paragraph("ОТЧЁТ О РИСКЕ БАНКРОТСТВА", title_style))
    story.append(Paragraph(company_name, heading_style))
    if inn:
        story.append(Paragraph(f"ИНН: {inn}", normal_style))
    story.append(Paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", normal_style))
    if industry_comparison:
        story.append(Paragraph(f"Отрасль: {industry_comparison.get('industry', 'Не указана')}", normal_style))
    story.append(Spacer(1, 25))

    story.append(Paragraph("Результат анализа", heading_style))
    table_data = [
        ["Модель:", model_choice],
        ["Риск:", res.get('risk', '—')],
        ["Вероятность банкротства:", f"{res.get('probability', 0):.1%}"],
        ["Score:", str(res.get('score', 0))]
    ]
    t = Table(table_data, colWidths=[170, 280])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    if industry_comparison:
        story.append(Paragraph("Сравнение с отраслевыми показателями", heading_style))
        industry_data = [["Показатель", "Компания", "Отрасль", "Разница"]]
        for metric_name, values in industry_comparison['metrics'].items():
            diff = values['difference']
            diff_str = f"{'+' if diff > 0 else ''}{diff:.3f}"
            industry_data.append([
                metric_name,
                str(values['company']),
                str(values['benchmark']),
                diff_str
            ])

        t_ind = Table(industry_data, colWidths=[150, 100, 100, 100])
        t_ind.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t_ind)
        story.append(Spacer(1, 20))

    story.append(Paragraph("Финансовый профиль", heading_style))
    radar_img = io.BytesIO()
    radar_fig.write_image(radar_img, format="png", scale=2.5)
    radar_img.seek(0)
    story.append(Image(radar_img, width=480, height=340))
    story.append(Spacer(1, 15))

    if imp_fig:
        story.append(Paragraph("Важность факторов", heading_style))
        imp_img = io.BytesIO()
        imp_fig.write_image(imp_img, format="png", scale=2.3)
        imp_img.seek(0)
        story.append(Image(imp_img, width=480, height=280))
        story.append(Spacer(1, 20))

    story.append(Paragraph("Рекомендации", heading_style))
    story.append(Paragraph(f"<b>{rec.get('urgency', '')}</b>", normal_style))
    if rec.get('main_problem'):
        story.append(Paragraph(f"<b>Главная проблема:</b> {rec['main_problem']}", normal_style))
    story.append(Paragraph("<b>Действия:</b>", normal_style))
    for action in rec.get('actions', []):
        story.append(Paragraph(f"• {action}", normal_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>План на 90 дней:</b>", normal_style))
    for i, step in enumerate(rec.get('plan_90_days', []), 1):
        story.append(Paragraph(f"{i}. {step}", normal_style))

    story.append(Spacer(1, 20))

    if forecast_df is not None and not forecast_df.empty:
        story.append(Paragraph("Прогноз на 1-3 года", heading_style))

        desired_order = ['Сценарий', 'Год', 'Выручка (млн)', 'EBIT (млн)',
                         'Активы (млн)', 'Долг (млн)', 'Чистая прибыль (млн)',
                         'WC/TA', 'TL/TA', 'EBIT/TA', 'Вероятность', 'Риск']

        available_cols = [col for col in desired_order if col in forecast_df.columns]
        forecast_display = forecast_df[available_cols].copy()

        forecast_data = [forecast_display.columns.tolist()]
        for _, row in forecast_display.iterrows():
            forecast_data.append([str(val) for val in row.values])

        t_forecast = Table(forecast_data, repeatRows=1)
        t_forecast.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(t_forecast)

    story.append(Spacer(1, 30))
    story.append(Paragraph("Отчёт подготовлен системой <b>Прогнозирование банкроства</b>", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


st.set_page_config(page_title="Прогнозирование банкроства", layout="wide", page_icon="📊")

conn = sqlite3.connect('reports.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
c.execute(
    '''CREATE TABLE IF NOT EXISTS reports 
       (id INTEGER PRIMARY KEY, 
        username TEXT,
        timestamp TEXT, 
        company TEXT, 
        inn TEXT, 
        results TEXT,
        FOREIGN KEY (username) REFERENCES users (username))''')
conn.commit()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register():
    st.subheader("📝 Регистрация")
    new_user = st.text_input("Логин", key="reg_user")
    new_pass = st.text_input("Пароль", type="password", key="reg_pass")
    if st.button("Создать аккаунт", key="reg_btn"):
        if len(new_user.strip()) < 3 or len(new_pass.strip()) < 3:
            st.error("Логин и пароль должны содержать минимум 3 символа")
        else:
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                          (new_user, hash_password(new_pass)))
                conn.commit()
                st.success("✅ Аккаунт успешно создан!")
            except sqlite3.IntegrityError:
                st.error("Пользователь с таким логином уже существует")


def login():
    st.subheader("🔑 Вход")
    user = st.text_input("Логин", key="login_user")
    pw = st.text_input("Пароль", type="password", key="login_pw")
    if st.button("Войти", key="login_btn"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, hash_password(pw)))
        if c.fetchone():
            st.session_state.authenticated = True
            st.session_state.username = user
            st.rerun()
        else:
            st.error("❌ Неверный логин или пароль")


if not st.session_state.authenticated:
    st.title("📊 Прогнозирование банкроства")
    tab_login, tab_reg = st.tabs(["Вход", "Регистрация"])
    with tab_login:
        login()
    with tab_reg:
        register()
    st.stop()

if 'predictor' not in st.session_state:
    st.session_state.predictor = BankruptcyPredictor()
predictor = st.session_state.predictor

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

tab1, tab2, tab3, tab4 = st.tabs(["📊 Новый анализ", "⚖️ Сравнение компаний", "🧠 Обучение модели", "📚 История отчётов"])

with tab1:
    st.header("Новый анализ компании")

    sample_companies = {
        "7707083893": {
            "name": "ООО Газпром Межрегионгаз Иваново",
            "industry": "Торговля",
            "ta": 3850000000, "tl": 2680000000, "wc": 1120000000,
            "sales": 12850000000, "ebit": 480000000, "re": 1050000000,
            "pbt": 265000000, "mv": 4200000000, "bv": 2680000000,
            "cl": 1450000000, "ca": 2570000000, "equity": 1170000000,
            "cred_debt": 1820000000, "deb_debt": 1950000000,
            "loan": 980000000, "sales_profit": 520000000, "loss": 0
        },
        "7707033423": {
            "name": "ООО Эггер Древпродукт Шуя",
            "industry": "Производство",
            "ta": 14200000000, "tl": 5800000000, "wc": 3900000000,
            "sales": 13800000000, "ebit": 2150000000, "re": 4600000000,
            "pbt": 1980000000, "mv": 18500000000, "bv": 5800000000,
            "cl": 3200000000, "ca": 7100000000, "equity": 8400000000,
            "cred_debt": 2950000000, "deb_debt": 4120000000,
            "loan": 3100000000, "sales_profit": 2450000000, "loss": 0
        },
        "7712345678": {
            "name": "ООО ТехноПром",
            "industry": "Производство",
            "ta": 450000000, "tl": 280000000, "wc": 120000000,
            "sales": 920000000, "ebit": 85000000, "re": 210000000,
            "pbt": 78000000, "mv": 520000000, "bv": 170000000,
            "cl": 95000000, "ca": 215000000, "equity": 170000000,
            "cred_debt": 65000000, "deb_debt": 98000000,
            "loan": 145000000, "sales_profit": 92000000, "loss": 0
        }
    }

    col_load1, col_load2 = st.columns([3, 1])
    with col_load1:
        selected_inn = st.selectbox(
            "Быстрая загрузка компании",
            options=[""] + list(sample_companies.keys()),
            format_func=lambda x: f"{x} — {sample_companies[x]['name']}" if x else "- Выберите компанию -",
            key="quick_load_inn"
        )

    with col_load2:
        if st.button("Загрузить данные", type="secondary", key="load_btn"):
            if selected_inn and selected_inn in sample_companies:
                data = sample_companies[selected_inn]
                st.session_state.company_name = data['name']
                st.session_state.inn_input = selected_inn
                st.session_state.industry = data['industry']
                st.session_state.ta = float(data['ta'])
                st.session_state.wc = float(data['wc'])
                st.session_state.sales = float(data['sales'])
                st.session_state.ebit = float(data['ebit'])
                st.session_state.re = float(data['re'])
                st.session_state.pbt = float(data['pbt'])
                st.session_state.mv = float(data['mv'])
                st.session_state.bv = float(data['bv'])
                st.session_state.tl = float(data['tl'])
                st.session_state.cl = float(data['cl'])
                st.session_state.ca = float(data['ca'])
                st.session_state.equity = float(data['equity'])
                st.session_state.cred_debt = float(data['cred_debt'])
                st.session_state.deb_debt = float(data['deb_debt'])
                st.session_state.loan = float(data['loan'])
                st.session_state.sales_profit = float(data['sales_profit'])
                st.session_state.loss = float(data['loss'])

                st.success(f"✅ Загружены данные: {data['name']}")
                st.rerun()

    inn = st.text_input("ИНН", placeholder="7707083893", key="inn_input")

    company_name = st.text_input("Название организации",
                                 key="company_name")

    industry = st.selectbox("Отрасль",
                            ["Строительство", "Торговля", "Производство", "IT и услуги",
                             "Сельское хозяйство", "Транспорт", "Другая"], key ="industry")

    col1, col2, col3 = st.columns(3)
    with col1:
        ta = st.number_input("Общие активы (TA)", key="ta")
        wc = st.number_input("Оборотный капитал (WC)", key="wc")
        sales = st.number_input("Выручка (Sales)", key="sales")
        ebit = st.number_input("EBIT", key="ebit")
    with col2:
        re = st.number_input("Нераспределённая прибыль (RE)", key="re")
        pbt = st.number_input("Прибыль до налогов (PBT)", key="pbt")
        mv = st.number_input("Рыночная стоимость (MV)", key="mv")
        bv = st.number_input("Балансовая стоимость (BV)", key="bv")
    with col3:
        tl = st.number_input("Общие обязательства (TL)", key="tl")
        cl = st.number_input("Текущие обязательства (CL)", key="cl")
        ca = st.number_input("Текущие активы (CA)", key="ca")

    with st.expander("Дополнительные поля для модели Зайцевой"):
        loss = st.number_input("Чистый убыток", key="loss")
        equity = st.number_input("Собственный капитал", key="equity")
        cred_debt = st.number_input("Кредиторская задолженность", key="cred_debt")
        deb_debt = st.number_input("Дебиторская задолженность", key="deb_debt")
        loan = st.number_input("Заёмные средства", key="loan")
        sales_profit = st.number_input("Прибыль от продаж", key="sales_profit")

    model_choice = st.selectbox("Выберите модель", [
        "Альтмана", "Спрингейта", "Таффлера", "Зайцевой", "AI (TensorFlow Neural Network)"
    ])

    with st.expander("📈 Прогноз на 1–3 года", expanded=True):
        forecast_years = st.slider("Горизонт прогноза (лет)", 1, 3, 3)

    if st.button("Рассчитать", type="primary"):
        ml_features = [wc, ta, re, ebit, mv, bv, sales, tl, pbt, cl, ca,
                       loss, equity, cred_debt, deb_debt, loan, sales_profit]

        current_data = {
            'ta': ta, 'wc': wc, 'ebit': ebit, 'tl': tl, 'sales': sales,
            're': re, 'pbt': pbt, 'cl': cl, 'ca': ca, 'equity': equity,
            'cred_debt': cred_debt, 'deb_debt': deb_debt, 'loan': loan
        }

        if model_choice == "Альтмана":
            res = predictor.altman_z_score(wc, ta, re, ebit, mv, bv, sales)
        elif model_choice == "Спрингейта":
            res = predictor.springate_model(wc, ta, ebit, tl, sales)
        elif model_choice == "Таффлера":
            res = predictor.taffler_model(pbt, cl, ca, tl, sales, ta)
        elif model_choice == "Зайцевой":
            res = predictor.zaitseva_model(loss, equity, cred_debt, deb_debt, sales, ta, loan, sales_profit)
        else:
            res = predictor.predict_bankruptcy_ml(ml_features)

        industry_comparison = predictor.compare_with_industry(current_data, industry)

        importance_dict = predictor.calculate_feature_importance(ml_features, model_choice, current_data)

        forecast_opt = predictor.forecast_bankruptcy_improved(current_data, years=forecast_years, scenario="Оптимистичный")
        forecast_base = predictor.forecast_bankruptcy_improved(current_data, years=forecast_years, scenario="Базовый")
        forecast_pes = predictor.forecast_bankruptcy_improved(current_data, years=forecast_years, scenario="Пессимистичный")

        forecast_df = pd.concat([forecast_opt, forecast_base, forecast_pes], ignore_index=True)

        rec = predictor.generate_recommendations(res, model_choice, current_data, importance_dict)

        radar_fig = predictor.create_radar_chart(current_data, company_name)
        imp_fig = predictor.create_feature_importance_chart(importance_dict, model_choice)

        st.session_state.analysis_results = {
            'company_name': company_name,
            'inn': inn,
            'model_choice': model_choice,
            'res': res,
            'current_data': current_data,
            'forecast_df': forecast_df,
            'rec': rec,
            'radar_fig': radar_fig,
            'imp_fig': imp_fig,
            'importance_dict': importance_dict,
            'industry_comparison': industry_comparison
        }

        report_data = {
            'company_name': company_name,
            'inn': inn,
            'model_choice': model_choice,
            'risk': res['risk'],
            'probability': res['probability'],
            'industry': industry
        }

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute(
            "INSERT INTO reports (username, timestamp, company, inn, results) VALUES (?, ?, ?, ?, ?)",
            (st.session_state.username, timestamp, company_name, inn, json.dumps(report_data, ensure_ascii=False))
        )
        conn.commit()

        st.success("✅ Анализ выполнен и сохранён в историю!")

    if st.session_state.analysis_results:
        data = st.session_state.analysis_results
        res = data['res']
        forecast_df = data['forecast_df']
        rec = data['rec']
        radar_fig = data['radar_fig']
        imp_fig = data['imp_fig']
        industry_comparison = data['industry_comparison']

        st.subheader(f"Результат: **{res['risk']}**")
        st.metric("Вероятность банкротства", f"{res['probability']:.1%}", delta=f"Score: {res['score']}")

        st.markdown("---")
        st.subheader(f"📊 Сравнение с отраслью: {industry_comparison['industry']}")

        industry_df = pd.DataFrame([
            {
                "Показатель": name,
                "Компания": val["company"],
                "Отрасль": val["benchmark"],
                "Разница": f"{'+' if val['difference'] > 0 else ''}{val['difference']:.3f}",
                "Статус": "✅ Выше среднего" if val["better"] else "⚠️ Ниже среднего"
            }
            for name, val in industry_comparison["metrics"].items()
        ])

        st.dataframe(industry_df, use_container_width=True, hide_index=True)

        fig_industry = go.Figure()
        metrics_names = list(industry_comparison["metrics"].keys())
        company_values = [industry_comparison["metrics"][m]["company"] for m in metrics_names]
        industry_values = [industry_comparison["metrics"][m]["benchmark"] for m in metrics_names]

        fig_industry.add_trace(go.Bar(
            name='Компания',
            x=metrics_names,
            y=company_values,
            marker_color='#1E88E5'
        ))
        fig_industry.add_trace(go.Bar(
            name='Отрасль',
            x=metrics_names,
            y=industry_values,
            marker_color='#FF2807'
        ))

        fig_industry.update_layout(
            title="Сравнение с отраслевыми показателями",
            barmode='group',
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_industry, use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(radar_fig, use_container_width=True)
        with col_g2:
            st.plotly_chart(imp_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Рекомендации")
        st.markdown(f"**{rec['urgency']}**")
        if rec.get('main_problem'):
            st.info(f"**Главная проблема:** {rec['main_problem']}")
        for action in rec.get('actions', []):
            st.write(f"• {action}")

        st.subheader("План на 90 дней")
        for i, step in enumerate(rec.get('plan_90_days', []), 1):
            st.write(f"{i}. {step}")

        st.markdown("---")
        st.subheader("Прогноз на 1–3 года")

        for scenario in ["Оптимистичный", "Базовый", "Пессимистичный"]:
            scenario_df = forecast_df[forecast_df['Сценарий'] == scenario]
            st.markdown(f"**Сценарий: {scenario}**")
            st.dataframe(scenario_df, use_container_width=True)
            st.markdown("")

        pdf_buffer = generate_pdf_report(
            data['company_name'],
            data['inn'],
            data['model_choice'],
            data['res'],
            data['current_data'],
            data['forecast_df'],
            data['rec'],
            data['radar_fig'],
            data['imp_fig'],
            data['industry_comparison']
        )

        st.download_button(
            label="📄 Скачать PDF отчёт",
            data=pdf_buffer,
            file_name=f"Банкротство_{data['company_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary"
        )

with tab2:
    st.header("⚖️ Сравнение компаний")

    model_choice_compare = st.selectbox("Выберите модель для сравнения", [
        "Альтмана", "Спрингейта", "Таффлера", "Зайцевой", "AI (TensorFlow Neural Network)"
    ], key="compare_model")

    industry_compare = st.selectbox("Отрасль для сравнения",
                                    ["Строительство", "Торговля", "Производство", "IT и услуги", "Сельское хозяйство",
                                     "Транспорт", "Другая"], key="compare_industry")

    sample_companies = {
        "7707083893": {"name": "ООО Газпром Межрегионгаз Иваново", "ta": 3850000000, "tl": 2680000000, "wc": 1120000000,
                       "sales": 12850000000, "ebit": 480000000, "re": 1050000000, "pbt": 265000000, "mv": 4200000000,
                       "bv": 2680000000, "cl": 1450000000, "ca": 2570000000, "equity": 1170000000,
                       "cred_debt": 1820000000,
                       "deb_debt": 1950000000, "loan": 980000000, "sales_profit": 520000000, "loss": 0},
        "7707033423": {
            "name": "ООО Эггер Древпродукт Шуя",
            "industry": "Производство",
            "ta": 14200000000, "tl": 5800000000, "wc": 3900000000,
            "sales": 13800000000, "ebit": 2150000000, "re": 4600000000,
            "pbt": 1980000000, "mv": 18500000000, "bv": 5800000000,
            "cl": 3200000000, "ca": 7100000000, "equity": 8400000000,
            "cred_debt": 2950000000, "deb_debt": 4120000000,
            "loan": 3100000000, "sales_profit": 2450000000, "loss": 0
        },
        "7712345678": {"name": "ООО ТехноПром", "ta": 450000000, "tl": 280000000, "wc": 120000000,
                       "sales": 920000000, "ebit": 85000000, "re": 210000000, "pbt": 78000000, "mv": 520000000,
                       "bv": 170000000, "cl": 95000000, "ca": 215000000, "equity": 170000000, "cred_debt": 65000000,
                       "deb_debt": 98000000, "loan": 145000000, "sales_profit": 92000000, "loss": 0}
    }

    selected_inns = st.multiselect(
        "Выберите компании (до 3)",
        options=list(sample_companies.keys()),
        format_func=lambda x: f"{x} — {sample_companies[x]['name']}",
        max_selections=3
    )

    forecast_years_compare = st.slider("Горизонт прогноза (лет)", 1, 3, 3, key="forecast_compare")

    if st.button("Сравнить компании", type="primary") and selected_inns:
        cols = st.columns(len(selected_inns))

        for idx, inn_val in enumerate(selected_inns):
            data = sample_companies[inn_val]
            company_name = data["name"]

            with cols[idx]:
                st.subheader(company_name)
                st.caption(f"ИНН: {inn_val}")

                ml_features = [
                    data['wc'], data['ta'], data['re'], data['ebit'],
                    data.get('mv', data['ta'] * 1.2), data.get('bv', data['ta'] * 0.6),
                    data['sales'], data['tl'], data['pbt'], data['cl'], data['ca'],
                    data['loss'], data['equity'], data['cred_debt'], data['deb_debt'],
                    data['loan'], data['sales_profit']
                ]

                current_data = {
                    'ta': data['ta'], 'wc': data['wc'], 'ebit': data['ebit'],
                    'tl': data['tl'], 'sales': data['sales'],
                    're': data['re'], 'pbt': data['pbt'], 'cl': data['cl'],
                    'ca': data['ca'], 'equity': data['equity'],
                    'cred_debt': data['cred_debt'], 'deb_debt': data['deb_debt'],
                    'loan': data['loan'], 'loss': data.get('loss', 0)
                }

                if model_choice_compare == "Альтмана":
                    res = predictor.altman_z_score(data['wc'], data['ta'], data['re'], data['ebit'],
                                                   data.get('mv', data['ta'] * 1.2),
                                                   data.get('bv', data['ta'] * 0.6),
                                                   data['sales'])
                elif model_choice_compare == "Спрингейта":
                    res = predictor.springate_model(data['wc'], data['ta'], data['ebit'],
                                                    data['tl'], data['sales'])
                elif model_choice_compare == "Таффлера":
                    res = predictor.taffler_model(data['pbt'], data['cl'], data['ca'],
                                                  data['tl'], data['sales'], data['ta'])
                elif model_choice_compare == "Зайцевой":
                    res = predictor.zaitseva_model(data['loss'], data['equity'], data['cred_debt'],
                                                   data['deb_debt'], data['sales'], data['ta'],
                                                   data['loan'], data['sales_profit'])
                else:
                    res = predictor.predict_bankruptcy_ml(ml_features)

                st.metric("Вероятность банкротства", f"{res['probability']:.1%}",
                          delta=f"Score: {res['score']}")
                if "Низкий" in res['risk']:
                    st.success(res['risk'])
                elif "Серая" in res['risk'] or "Средний" in res['risk']:
                    st.warning(res['risk'])
                else:
                    st.error(res['risk'])

                st.markdown("---")
                st.markdown("**📊 Сравнение с отраслью**")
                industry_comparison = predictor.compare_with_industry(current_data, industry_compare)

                industry_df = pd.DataFrame([
                    {
                        "Показатель": name,
                        "Компания": val["company"],
                        "Отрасль": val["benchmark"],
                        "Разница": f"{'+' if val['difference'] > 0 else ''}{val['difference']:.3f}",
                        "Статус": "✅ Выше" if val["better"] else "⚠️ Ниже"
                    }
                    for name, val in industry_comparison["metrics"].items()
                ])

                st.dataframe(industry_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                fig_radar = predictor.create_radar_chart(data, company_name)
                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_compare_{inn_val}_{idx}")

                importance_dict = predictor.calculate_feature_importance(
                    ml_features, model_choice_compare, current_data
                )
                fig_imp = predictor.create_feature_importance_chart(importance_dict, model_choice_compare)
                st.plotly_chart(fig_imp, use_container_width=True, key=f"imp_compare_{inn_val}_{idx}")

                st.markdown("---")
                st.markdown("**📈 Прогноз**")

                forecast_opt = predictor.forecast_bankruptcy_improved(
                    current_data, years=forecast_years_compare, scenario="Оптимистичный"
                )
                forecast_base = predictor.forecast_bankruptcy_improved(
                    current_data, years=forecast_years_compare, scenario="Базовый"
                )
                forecast_pes = predictor.forecast_bankruptcy_improved(
                    current_data, years=forecast_years_compare, scenario="Пессимистичный"
                )

                forecast_df = pd.concat([forecast_opt, forecast_base, forecast_pes], ignore_index=True)

                for scenario in ["Оптимистичный", "Базовый", "Пессимистичный"]:
                    scenario_df = forecast_df[forecast_df['Сценарий'] == scenario]
                    st.markdown(f"*{scenario}*")
                    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("**💡 Рекомендации**")

                rec = predictor.generate_recommendations(
                    res, model_choice_compare, current_data, importance_dict
                )

                st.markdown(f"**{rec['urgency']}**")
                if rec.get('main_problem'):
                    st.info(f"**Главная проблема:** {rec['main_problem']}")

                st.markdown("*Действия:*")
                for action in rec.get('actions', []):
                    st.write(f"• {action}")

                st.markdown("*План на 90 дней:*")
                for i, step in enumerate(rec.get('plan_90_days', []), 1):
                    st.write(f"{i}. {step}")

                st.markdown("---")
                try:
                    pdf_buffer = generate_pdf_report(
                        company_name, inn_val, model_choice_compare,
                        res, current_data, forecast_df, rec,
                        fig_radar, fig_imp, industry_comparison
                    )

                    st.download_button(
                        label="📄 Скачать PDF",
                        data=pdf_buffer,
                        file_name=f"Банкротство_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_compare_{inn_val}_{idx}"
                    )
                except Exception as e:
                    st.warning(f"PDF не сгенерирован: {e}")

with tab3:
    st.header("🧠 Обучение нейросети")

    uploaded = st.file_uploader("Загрузите CSV датасет (18 колонок)", type="csv")

    if uploaded and st.button("Начать обучение", type="primary"):
        try:
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_path = os.path.join(tmpdirname, "temp_train.csv")

                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                plot_ph = st.empty()
                with st.spinner("Обучение модели... Это может занять 10–30 секунд"):
                    success = predictor.train_model(temp_path, plot_ph)

                if success:
                    st.success("✅ Модель успешно обучена и сохранена!")

        except Exception as e:
            st.error(f"Ошибка при обучении: {e}")

with tab4:
    st.header("📚 История отчётов")

    col1, col2 = st.columns([3, 1])
    with col1:
        df_hist = pd.read_sql_query(
            "SELECT * FROM reports WHERE username = ? ORDER BY id DESC LIMIT 20",
            conn,
            params=(st.session_state.username,)
        )
    with col2:
        if st.button("🗑️ Очистить мою историю", type="secondary"):
            c.execute("DELETE FROM reports WHERE username = ?", (st.session_state.username,))
            conn.commit()
            st.success("История очищена!")
            st.rerun()

    if not df_hist.empty:
        st.info(f"Показаны отчёты пользователя: **{st.session_state.username}**")
        for idx, row in df_hist.iterrows():
            with st.expander(f"{row['timestamp']} — {row['company']} (ИНН: {row['inn']})"):
                try:
                    results = json.loads(row['results'])
                    st.write(f"**Модель:** {results.get('model_choice', 'Н/Д')}")
                    st.write(f"**Риск:** {results.get('risk', 'Н/Д')}")
                    st.write(f"**Вероятность:** {results.get('probability', 0):.1%}")
                    st.write(f"**Отрасль:** {results.get('industry', 'Н/Д')}")
                except:
                    st.write("Данные отчёта повреждены")
    else:
        st.info("История отчётов пуста. Выполните анализ компании, чтобы сохранить результат.")
