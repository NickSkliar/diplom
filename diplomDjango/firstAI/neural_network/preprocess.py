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
        self.data_type = data_type  # 'train' или 'test'
        self.chunk_size = chunk_size  # Размер чанка для обработки данных

        # Сопоставление атрибутов модели Weight с полями в таблице данных
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
            # Добавьте дополнительные сопоставления, если необходимо
        }

        # Получение списка активных признаков и их весов из модели Weight
        active_weights = Weight.objects.filter(is_active=True)

        self.db_features = []
        self.features = []
        self.feature_weights = []

        for weight in active_weights:
            feature = self.attribute_mapping[weight.attribute]
            if feature == 'timestamp':
                # Разбиваем 'timestamp' на 'hour' и 'day_of_week'
                self.features.extend(['hour', 'day_of_week'])
                self.db_features.append('timestamp')  # Для запроса к базе данных
                # Присваиваем один и тот же вес обоим новым признакам
                self.feature_weights.extend([weight.weight, weight.weight])
            else:
                self.features.append(feature)
                self.db_features.append(feature)
                self.feature_weights.append(weight.weight)

        # Проверяем соответствие между количеством признаков и весов
        if len(self.features) != len(self.feature_weights):
            raise ValueError(f"Несоответствие размерностей: количество признаков ({len(self.features)}) и количество весов ({len(self.feature_weights)}) не совпадают.")

        # Инициализация StandardScaler
        if self.data_type == 'train':
            self.scaler = StandardScaler()
            self.fit_scaler()
        else:
            # Загружаем обученный scaler
            scaler_path = os.path.join('models', 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"Масштабировщик загружен из '{scaler_path}'.")
            else:
                raise FileNotFoundError(f"Не найден файл масштабировщика '{scaler_path}'. Пожалуйста, сначала обучите модель.")

    def fit_scaler(self):
        # Загрузка всех данных для обучения масштабировщика
        queryset = DownloadDB.objects.filter(data_type='train')
        records = []
        total_records = queryset.count()
        if total_records == 0:
            print("Нет данных для обучения масштабировщика.")
            return

        for batch_start in range(0, total_records, self.chunk_size):
            batch_queryset = queryset[batch_start:batch_start+self.chunk_size]
            batch_records = list(batch_queryset.values(*self.db_features))
            batch_df = pd.DataFrame.from_records(batch_records)
            batch_df = self.preprocess_batch(batch_df, scale_data=False)
            if not batch_df.empty:
                records.append(batch_df)
            else:
                print("После предобработки данные отсутствуют в текущем чанке при обучении масштабировщика.")

        if records:
            full_data = pd.concat(records, ignore_index=True)
            self.scaler.fit(full_data[self.features])
        else:
            print("Нет данных для обучения масштабировщика.")

    def get_data(self):
        # Загрузка данных по чанкам
        queryset = DownloadDB.objects.filter(data_type=self.data_type)
        total_records = queryset.count()
        if total_records == 0:
            print(f"Нет данных типа {self.data_type}.")
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
                print("После предобработки данные отсутствуют в текущем чанке.")

    def preprocess_batch(self, data, scale_data=True):
        # Удаление строк с пропусками
        data = data.dropna()

        # Проверяем, есть ли колонка 'label' в данных
        if self.label_column in data.columns:
            # Кодирование меток
            label_mapping = {
                'Benign': 0,
                'ddos': 1
                # Добавьте дополнительные метки, если необходимо
            }
            data[self.label_column] = data[self.label_column].map(label_mapping)
            # Удаление строк с неизвестными метками
            data = data.dropna(subset=[self.label_column])

        # Обработка timestamp
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data = data.drop(columns=['timestamp'], errors='ignore')

        # Преобразование IP-адресов в числовой формат
        if 'src_ip' in data.columns:
            data['src_ip'] = data['src_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
        if 'dst_ip' in data.columns:
            data['dst_ip'] = data['dst_ip'].apply(lambda x: int(ipaddress.ip_address(x)))

        # Обработка бесконечностей и слишком больших значений
        for feature in self.features:
            if feature in data.columns:
                data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
                data[feature] = data[feature].apply(lambda x: np.nan if abs(x) > 1e10 else x)

        # Приведение данных к числовому типу и обработка ошибок
        for feature in self.features:
            if feature in data.columns:
                data[feature] = pd.to_numeric(data[feature], errors='coerce')

        # Удаление строк с NaN после всех преобразований
        initial_row_count = len(data)
        data = data.dropna()
        removed_rows = initial_row_count - len(data)

        if removed_rows > 0:
            print(f"Удалено {removed_rows} строк с некорректными значениями.")

        # Убедимся, что все необходимые признаки присутствуют
        for feature in self.features:
            if feature not in data.columns:
                print(f"Признак {feature} отсутствует в данных, заполняем нулями.")
                data[feature] = 0.0

        # Масштабирование признаков
        if scale_data and not data.empty:
            data[self.features] = self.scaler.transform(data[self.features])

        # Приведение типов данных к float32
        data[self.features] = data[self.features].astype(np.float32)
        if self.label_column in data.columns:
            data[self.label_column] = data[self.label_column].astype(np.float32)

        return data

    def get_feature_weights(self):
        return self.feature_weights

    def get_input_size(self):
        return len(self.features)
