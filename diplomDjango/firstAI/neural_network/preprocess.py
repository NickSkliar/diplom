# firstAI/neural_network/preprocess.py

import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler
from database.models import DownloadDB
from firstAI.models import Weight
import joblib
import os

class DataLoader:
    def __init__(self, data_type='train', batch_size=10000, chunk_size=50000):
        self.batch_size = batch_size
        self.label_column = 'label'
        self.data_type = data_type  # 'train' або 'test'
        self.chunk_size = chunk_size  # Розмір чанка для обробки даних

        # Відповідність атрибутів моделі Weight з полями в таблиці даних
        self.attribute_mapping = {
            'src_ip': 'src_ip',
            'dst_ip': 'dst_ip',
            'src_port': 'src_port',
            'dst_port': 'dst_port',
            'timestamp': 'timestamp',
            'flow_duration': 'flow_duration',
            'tot_fwd_pkts': 'tot_fwd_pkts',
            'tot_bwd_pkts': 'tot_bwd_pkts',
            'flow_byts_per_sec': 'flow_byts_per_sec',
            'flow_pkts_per_sec': 'flow_pkts_per_sec',
            'fwd_iat_mean': 'fwd_iat_mean',
            'bwd_iat_mean': 'bwd_iat_mean',
            'protocol': 'protocol',
            'down_up_ratio': 'down_up_ratio',
            'pkt_size_avg': 'pkt_size_avg',
            'fwd_pkts_per_sec': 'fwd_pkts_per_sec',
            'bwd_pkts_per_sec': 'bwd_pkts_per_sec',
            # Додайте додаткові відповідності за потреби
        }

        # Отримання списку активних ознак і їх ваг з моделі Weight
        active_weights = Weight.objects.filter(is_active=True)

        self.db_features = []
        self.features = []
        self.feature_weights = []

        for weight in active_weights:
            feature = self.attribute_mapping[weight.attribute]
            if feature == 'timestamp':
                # Розбиття 'timestamp' на 'hour' та 'day_of_week'
                self.features.extend(['hour', 'day_of_week'])
                self.db_features.append('timestamp')  # Для запиту до бази даних
                # Призначаємо однакову вагу обом новим ознакам
                self.feature_weights.extend([weight.weight, weight.weight])
            else:
                self.features.append(feature)
                self.db_features.append(feature)
                self.feature_weights.append(weight.weight)

        # Перевіряємо відповідність між кількістю ознак та ваг
        if len(self.features) != len(self.feature_weights):
            raise ValueError(f"Несумісність розмірностей: кількість ознак ({len(self.features)}) та кількість ваг ({len(self.feature_weights)}) не збігаються.")

        # Ініціалізація StandardScaler
        if self.data_type == 'train':
            self.scaler = StandardScaler()
            self.fit_scaler()
        else:
            # Завантажуємо навчений масштабувальник
            scaler_path = os.path.join('models', 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"Масштабувальник завантажено з '{scaler_path}'.")
            else:
                raise FileNotFoundError(f"Не знайдено файл масштабувальника '{scaler_path}'. Спочатку навчіть модель.")

    def fit_scaler(self):
        # Завантаження всіх даних для навчання масштабувальника
        queryset = DownloadDB.objects.filter(data_type='train')
        records = []
        total_records = queryset.count()
        if total_records == 0:
            print("Немає даних для навчання масштабувальника.")
            return

        for batch_start in range(0, total_records, self.chunk_size):
            batch_queryset = queryset[batch_start:batch_start+self.chunk_size]
            batch_records = list(batch_queryset.values(*self.db_features))
            batch_df = pd.DataFrame.from_records(batch_records)
            batch_df = self.preprocess_batch(batch_df, scale_data=False)
            if not batch_df.empty:
                records.append(batch_df)
            else:
                print("Після передобробки дані відсутні в поточному чанку під час навчання масштабувальника.")

        if records:
            full_data = pd.concat(records, ignore_index=True)
            self.scaler.fit(full_data[self.features])
        else:
            print("Немає даних для навчання масштабувальника.")

    def get_data(self):
        # Завантаження даних по чанках
        queryset = DownloadDB.objects.filter(data_type=self.data_type)
        total_records = queryset.count()
        if total_records == 0:
            print(f"Немає даних типу {self.data_type}.")
            return

        for batch_start in range(0, total_records, self.chunk_size):
            batch_queryset = queryset[batch_start:batch_start+self.chunk_size]
            batch_records = list(batch_queryset.values(*self.db_features, self.label_column))
            batch_df = pd.DataFrame.from_records(batch_records)
            batch_df = self.preprocess_batch(batch_df, scale_data=True)
            if not batch_df.empty:
                X = batch_df[self.features].values
                y = batch_df[self.label_column].values
                yield X, y
            else:
                print("Після передобробки дані відсутні в поточному чанку.")

    def preprocess_batch(self, data, scale_data=True):
        # Видалення рядків з пропусками
        data = data.dropna()

        # Перевіряємо, чи є стовпець 'label' у даних
        if self.label_column in data.columns:
            # Кодування міток
            label_mapping = {
                'Benign': 0,
                'ddos': 1
            }
            data[self.label_column] = data[self.label_column].map(label_mapping)
            # Видалення рядків з невідомими мітками
            data = data.dropna(subset=[self.label_column])

        # Обробка timestamp
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data = data.drop(columns=['timestamp'], errors='ignore')

        # Перетворення IP-адрес у числовий формат
        if 'src_ip' in data.columns:
            data['src_ip'] = data['src_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
        if 'dst_ip' in data.columns:
            data['dst_ip'] = data['dst_ip'].apply(lambda x: int(ipaddress.ip_address(x)))

        # Обробка нескінченностей і надто великих значень
        for feature in self.features:
            if feature in data.columns:
                data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
                data[feature] = data[feature].apply(lambda x: np.nan if abs(x) > 1e10 else x)

        # Приведення даних до числового типу та обробка помилок
        for feature in self.features:
            if feature in data.columns:
                data[feature] = pd.to_numeric(data[feature], errors='coerce')

        # Видалення рядків з NaN після всіх перетворень
        initial_row_count = len(data)
        data = data.dropna()
        removed_rows = initial_row_count - len(data)

        if removed_rows > 0:
            print(f"Видалено {removed_rows} рядків з некоректними значеннями.")

        # Переконаємося, що всі необхідні ознаки присутні
        for feature in self.features:
            if feature not in data.columns:
                print(f"Ознака {feature} відсутня в даних, заповнюємо нулями.")
                data[feature] = 0.0

        # Масштабування ознак
        if scale_data and not data.empty:
            data[self.features] = self.scaler.transform(data[self.features])

        # Приведення типів даних до float32
        data[self.features] = data[self.features].astype(np.float32)
        if self.label_column in data.columns:
            data[self.label_column] = data[self.label_column].astype(np.float32)

        return data

    def get_feature_weights(self):
        return self.feature_weights

    def get_input_size(self):
        return len(self.features)
