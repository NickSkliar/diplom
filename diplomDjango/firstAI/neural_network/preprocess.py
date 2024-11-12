# preprocess.py

import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler
from database.models import TrainingData, TestingData
from firstAI.models import Weight
import joblib

class DataLoader:
    def __init__(self, data_source='training', batch_size=10000):
        """
        Инициализация DataLoader.
        :param data_source: 'training' или 'testing', указывает, откуда загружаются данные.
        :param batch_size: Размер пакета данных для обработки.
        """
        self.batch_size = batch_size
        self.label_column = 'label'
        self.scaler = StandardScaler()
        self.data_source = data_source  # Указание источника данных

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

        # Получение списка активных признаков из модели Weight
        active_weights = Weight.objects.filter(is_active=True)
        self.features = [self.attribute_mapping[weight.attribute] for weight in active_weights]

    def load_data(self, limit=None):
        """
        Загрузка данных из базы данных.
        :param limit: Ограничение на количество загружаемых строк.
        :return: DataFrame с данными.
        """
        # Выбор источника данных
        if self.data_source == 'training':
            queryset = TrainingData.objects.all()
        elif self.data_source == 'testing':
            queryset = TestingData.objects.all()
        else:
            raise ValueError("data_source должен быть 'training' или 'testing'")

        if limit:
            queryset = queryset[:limit]

        # Преобразование данных в DataFrame
        data = pd.DataFrame.from_records(queryset.values(*self.features, self.label_column))
        return data

    def preprocess_data(self, data):
        """
        Предобработка данных: обработка некорректных значений, масштабирование, обработка timestamp и IP.
        :param data: DataFrame с исходными данными.
        :return: DataFrame с обработанными данными.
        """
        # Удаление строк с пропусками
        data = data.dropna()

        # Кодирование меток
        label_mapping = {
            'Benign': 0,
            'ddos': 1
            # Добавьте дополнительные метки, если необходимо
        }
        data[self.label_column] = data[self.label_column].map(label_mapping)

        # Удаление строк с неизвестными метками
        data = data.dropna(subset=[self.label_column])

        # Обработка timestamp, если он есть среди признаков
        if 'timestamp' in self.features:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data = data.drop(columns=['timestamp'], errors='ignore')

            # Обновляем список признаков
            self.features = [feature for feature in self.features if feature != 'timestamp']
            self.features.extend(['hour', 'day_of_week'])

        # Преобразование IP-адресов в числовой формат
        if 'src_ip' in self.features:
            data['src_ip'] = data['src_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
        if 'dst_ip' in self.features:
            data['dst_ip'] = data['dst_ip'].apply(lambda x: int(ipaddress.ip_address(x)))

        # Обработка бесконечностей и слишком больших значений
        for feature in self.features:
            if feature in data.columns:
                # Заменяем бесконечности на NaN
                data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
                # Убираем слишком большие значения
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
            print(f"Удалено {removed_rows} строк с некорректными значениями (NaN, бесконечности, слишком большие числа).")

        # Масштабирование признаков
        data[self.features] = self.scaler.fit_transform(data[self.features])

        # Приведение типов данных к float32
        data[self.features] = data[self.features].astype(np.float32)
        data[self.label_column] = data[self.label_column].astype(np.float32)

        return data

    def get_data(self, limit=None):
        """
        Загрузка и предобработка данных.
        :param limit: Ограничение на количество строк.
        :return: Массивы X и y.
        """
        data = self.load_data(limit)
        data = self.preprocess_data(data)
        X = data[self.features].values
        y = data[self.label_column].values
        return X, y

    def save_scaler(self, path):
        """
        Сохранение масштабировщика для дальнейшего использования.
        :param path: Путь для сохранения.
        """
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        """
        Загрузка сохранённого масштабировщика.
        :param path: Путь к сохранённому файлу.
        """
        self.scaler = joblib.load(path)

    def get_feature_weights(self):
        """
        Получение весов признаков из модели Weight.
        :return: Список весов.
        """
        active_weights = Weight.objects.filter(is_active=True)
        weights = {self.attribute_mapping[weight.attribute]: weight.weight for weight in active_weights}
        feature_weights = [weights.get(feature, 1.0) for feature in self.features]
        return feature_weights
