"""
Извлечение признаков из ЭЭГ данных
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


def extract_eeg_features(eeg_data: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Извлечение признаков из ЭЭГ данных
    
    Args:
        eeg_data: [C, T] - ЭЭГ сигнал (каналы, временные точки)
        fs: частота дискретизации (Гц)
    
    Returns:
        features: [n_features] - вектор признаков
    """
    n_channels, n_timepoints = eeg_data.shape
    
    freq_bands = {'Beta': (13, 30), 'Gamma': (30, 50)}
    
    # 1. Извлекаем признаки для каждого канала
    channel_features = []
    for ch_idx in range(n_channels):
        signal = eeg_data[ch_idx, :]
        
        # Частотные признаки
        frequencies, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(64, len(signal)))
        
        # Beta power
        idx_beta = np.logical_and(frequencies >= freq_bands['Beta'][0],
                                 frequencies <= freq_bands['Beta'][1])
        beta_power = np.trapz(psd[idx_beta], frequencies[idx_beta])
        
        # Gamma power
        idx_gamma = np.logical_and(frequencies >= freq_bands['Gamma'][0],
                                  frequencies <= freq_bands['Gamma'][1])
        gamma_power = np.trapz(psd[idx_gamma], frequencies[idx_gamma])
        
        # Спектральные признаки
        dominant_freq = frequencies[np.argmax(psd)]
        spectral_centroid = np.sum(frequencies * psd) / (np.sum(psd) + 1e-10)
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_idx = np.where(cumsum_psd >= 0.85 * total_energy)[0]
        spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
        spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid)**2) * psd) / (np.sum(psd) + 1e-10))
        
        # Статистические признаки
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        median_val = np.median(signal)
        variance_val = np.var(signal)
        energy_val = np.sum(signal**2)
        rms_val = np.sqrt(np.mean(signal**2))
        skewness_val = scipy_stats.skew(signal)
        kurtosis_val = scipy_stats.kurtosis(signal)
        
        # Hjorth Parameters
        activity = variance_val
        first_derivative = np.diff(signal)
        mobility = np.sqrt(np.var(first_derivative) / (np.var(signal) + 1e-10)) if len(first_derivative) > 0 else 0.0
        second_derivative = np.diff(first_derivative)
        if len(second_derivative) > 0 and np.var(first_derivative) > 1e-10 and mobility > 1e-10:
            complexity = np.sqrt(np.var(second_derivative) / (np.var(first_derivative) + 1e-10)) / mobility
        else:
            complexity = 0.0
        
        channel_features.append([
            beta_power, gamma_power,  # Частотные (2)
            mean_val, std_val, min_val, max_val, median_val,  # Базовые статистики (5)
            variance_val, energy_val, rms_val,  # Энергетические (3)
            skewness_val, kurtosis_val,  # Форма распределения (2)
            dominant_freq, spectral_centroid, spectral_rolloff, spectral_bandwidth,  # Спектральные (4)
            activity, mobility, complexity  # Hjorth Parameters (3)
        ])
    
    # 2. Вычисляем межканальные корреляции
    correlation_matrix = np.corrcoef(eeg_data)
    upper_triangle = np.triu(correlation_matrix, k=1)
    correlation_features = upper_triangle[upper_triangle != 0]
    
    # 3. Берем средние признаки по каналам (19 признаков)
    mean_channel_features = np.mean(channel_features, axis=0)
    
    # 4. Объединяем
    combined_features = np.concatenate([
        mean_channel_features,  # 19 признаков
        correlation_features    # n_channels*(n_channels-1)/2 признаков
    ])
    
    return combined_features.astype(np.float32)

