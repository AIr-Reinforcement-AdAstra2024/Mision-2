import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.signal import stft
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

plt.style.use('classic')


class SignalAnalytics:
    def __init__(self, data, threshold):
        """
        Initialize the class with the data file path and a threshold value.

        Parameters:
        data (str): Path to the CSV file containing the signal data.
        threshold (float): Threshold value for any analysis requiring it.
        """
        # Load the CSV file as a DataFrame
        self.data = pd.read_csv(data, skiprows=47, delimiter=";")
        self.threshold = threshold
        self.signal = None
        self.frecuency = self.data.iloc[:, 0].replace(',', '.', regex=True)
        self.magnitude = self.data.iloc[:, 1].replace(',', '.', regex=True)


        self.frecuency = self.frecuency.astype(float)
        self.magnitude = self.magnitude.astype(float)

    
        self.linear_mg = self.magnitude.apply(lambda x: 10**(x/20))
    

    """
    Main Frecuency
    """
    def frecuenciaCentral(self):
        
        # Compute the amplitude spectrum (absolute value of the FFT)
        amplitude = np.abs(self.linear_mg)

        # Find the index of the frequency with the highest amplitude
        idx_max = np.argmax(amplitude)  # Limit to positive frequencies

        # Get the frequency at the max amplitude
        dominant_freq = self.frecuency[idx_max]
        print(dominant_freq)

        return dominant_freq
    """
    Ancho banda
    """
    def anchoDeBanda(self, threshold_db=-3):
    

        # Compute the amplitude spectrum (absolute value of the FFT)
        amplitude = np.abs(self.linear_mg)  # Limit to positive frequencies

        # Power is proportional to the square of the amplitude
        power = amplitude**2

        # Find the maximum power value
        max_power = np.max(power)

        # Convert the threshold from dB to a linear scale
        threshold_power = max_power * (10 ** (threshold_db / 10))

        # Find frequencies where power is above the threshold
        indices_above_threshold = np.where(power >= threshold_power)[0]

        # If there are no frequencies above the threshold, return None
        if len(indices_above_threshold) == 0:
            return None

        # Find the lowest and highest frequencies within the threshold
        freq_low = self.frecuency[indices_above_threshold[0]]
        freq_high = self.frecuency[indices_above_threshold[-1]]

        # Bandwidth is the difference between the highest and lowest frequencies
        bandwidth = freq_high - freq_low
        print(bandwidth)

        return bandwidth, freq_low, freq_high
    
    
    """
    Potencia⑀
    """
    def potencia(self):
        """
        Calculates the power of a signal in milliwatts (mW).

        Parameters:
        - signal: array-like, the input signal (time domain, assumed in volts)
        - resistance: float, the load impedance (in ohms, default is 50 ohms)

        Returns:
        - power_mw: float, the average power of the signal in milliwatts
        """
        # Number of samples
        N = len(self.frecuency)
        magnitude_squared = np.abs(self.linear_mg) ** 2
        power = (1 / N) * np.sum(magnitude_squared)
        print(power)
        return power
    
    """
    SNR
    """
    def calcular_snr_sin_ruido(self):
        """
        Calcula el SNR (relación señal-ruido) estimado sin tener el ruido explícito.
        
        Parameters:
        - signal: array-like, la señal a analizar.
        
        Returns:
        - snr_db: float, el SNR estimado en decibelios (dB).
        """
        # Calcular la potencia promedio de la señal (mean squared value)
        signal_power = np.mean(np.square(self.linear_mg))
        
        # Estimar el ruido como la varianza de la señal
        noise_power = np.var(self.linear_mg)
        
        # Calcular el SNR en dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        print (snr_db)
        return snr_db
    
    """
    forma señal
    """
    def plot_fft_spectrum(fft_signal, sample_rate):
        """
        Plots the spectrum of an already FFT-transformed signal.

        Parameters:
        - fft_signal: array-like, the signal in the frequency domain (output of FFT).
        - sample_rate: int, the sampling rate of the original time-domain signal (in Hz).
        """
        # Number of points in the signal
        N = len(fft_signal)
        
        # Get the corresponding frequencies
        xf = fftfreq(N, 1 / sample_rate)
        
        # Compute the magnitude (amplitude) of the FFT signal
        amplitude = np.abs(fft_signal)
        
        # Limit to positive frequencies only
        xf = xf[:N // 2]
        amplitude = amplitude[:N // 2]
        
        # Plot the spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(xf, amplitude)
        plt.title("Signal Spectrum (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    """
    Frecuencias de espuria
    """
    def frecuenciasSpuria(self,threshold=0.1, frequency_range=10**6):
        """
        Detects spurious frequencies in the frequency spectrum.

        Parameters:
        - fft_signal: array-like, the signal in the frequency domain (output of FFT).
        - sample_rate: int, the sampling rate of the original time-domain signal (in Hz).
        - threshold: float, the relative amplitude below which spurious frequencies are considered (default=0.1).
        - frequency_range: float, the frequency range in Hz around the main signal to check for spurious frequencies (default=50).

        Returns:
        - spurious_frequencies: list of tuples, where each tuple is (frequency, amplitude).
        """
        
        main_signal_idx = np.argmax(self.linear_mg)
        main_frequency = self.frecuency[main_signal_idx]
        main_amplitude = self.linear_mg[main_signal_idx]
        
        # Threshold for detecting spurious frequencies
        spurious_threshold = main_amplitude * threshold
        
        # Find spurious frequencies near the main signal
        spurious_frequencies = []
        for i, amp in enumerate(self.linear_mg):
            freq = self.frecuency[i]
            # Check if the frequency is within the defined range and has significant amplitudud
            if (main_frequency - frequency_range <= freq <= main_frequency + frequency_range) and (amp < main_amplitude and amp > spurious_threshold):
                spurious_frequencies.append((freq, amp))
        print(spurious_frequencies)
        return spurious_frequencies
    
    """
    Frecuencias armónicas
    """
    def detect_harmonics_in_fft(fft_signal, sample_rate, fundamental_freq, 
                            threshold=0.1, tolerance=0.05):
        """
        Detects harmonic frequencies in a given FFT signal.

        Parameters:
        - fft_signal: array-like, the FFT of the signal.
        - sample_rate: int, the sampling rate of the original signal.
        - fundamental_freq: float, the known fundamental frequency (in Hz).
        - threshold: float, relative amplitude to consider as a harmonic.
        - tolerance: float, allowable difference between expected and detected 
        harmonics.

        Returns:
        - harmonic_frequencies: list of detected harmonic frequencies (in Hz).
        """
        # Number of points in the FFT signal
        N = len(fft_signal)
        
        # Calculate the frequencies for the FFT bins
        xf = np.fft.fftfreq(N, 1 / sample_rate)
        
        # Compute the magnitude of the FFT and limit to positive frequencies
        amplitude = np.abs(fft_signal[:N // 2])
        xf = xf[:N // 2]
        
        # Detect peaks in the FFT spectrum
        peaks, _ = find_peaks(amplitude, height=np.max(amplitude) * threshold)
        
        # Detect harmonics based on the fundamental frequency
        harmonic_frequencies = []
        for i in range(2, len(xf) // int(fundamental_freq)):
            harmonic_freq = fundamental_freq * i
            # Find the closest peak to the expected harmonic
            closest_peak = min(peaks, key=lambda x: abs(xf[x] - harmonic_freq))
            if abs(xf[closest_peak] - harmonic_freq) < fundamental_freq * tolerance:
                harmonic_frequencies.append(xf[closest_peak])
        
        return harmonic_frequencies
    
    
    """
    Interferencias
    """
    def eliminate_interference(fft_signal, sample_rate, fundamental_freq, threshold=0.1):
        """
        Eliminates interference signals from an FFT-transformed signal using `scipy.signal`.

        Parameters:
        - fft_signal: array-like, the input signal in the frequency domain (FFT).
        - sample_rate: int, the sampling rate of the original time-domain signal (in Hz).
        - fundamental_freq: float, the fundamental frequency to retain (in Hz).
        - threshold: float, relative amplitude threshold for detecting interference signals.

        Returns:
        - cleaned_fft_signal: the FFT signal with interference frequencies eliminated.
        """
        # Length of the signal
        N = len(fft_signal)
        
        # Compute the frequencies corresponding to the FFT bins
        xf = np.fft.fftfreq(N, 1 / sample_rate)
        
        # Compute the magnitude of the FFT signal
        amplitude = np.abs(fft_signal[:N // 2])  # Positive frequencies only
        xf = xf[:N // 2]  # Limit frequencies to positive side only
        
        # Detect peaks in the FFT amplitude spectrum
        peaks, properties = find_peaks(amplitude, height=np.max(amplitude) * threshold)
        
        # Find interference peaks (peaks that are not multiples of the fundamental frequency)
        interference_peaks = []
        for peak_idx in peaks:
            freq = xf[peak_idx]
            # Check if the frequency is not a harmonic of the fundamental frequency
            if not np.isclose(freq % fundamental_freq, 0, atol=fundamental_freq * 0.05):
                interference_peaks.append(peak_idx)
        
        # Copy the FFT signal for modification
        cleaned_fft_signal = fft_signal.copy()
        
        # Zero out the interference peaks
        for peak_idx in interference_peaks:
            cleaned_fft_signal[peak_idx] = 0
            # Since FFT is symmetric, zero the corresponding negative frequency as well
            cleaned_fft_signal[-peak_idx] = 0
        
        return cleaned_fft_signal
    
    """
    Modulación
    """
    def detect_modulation(signal, sample_rate):
        """
        Detects the modulation type of a given signal.
        
        Parameters:
        - signal: array-like, the input time-domain signal.
        - sample_rate: int, the sample rate of the signal (in Hz).

        Returns:
        - modulation_type: string, the detected modulation type (AM, FM, PM, QAM, or Unknown).
        """
        # Apply the Hilbert transform to extract the analytic signal (complex signal)
        analytic_signal = hilbert(signal)
        
        # Extract amplitude envelope (AM detection)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Extract instantaneous phase
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Extract instantaneous frequency (FM detection)
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
        
        # Check for Amplitude Modulation (AM): Look for variations in the amplitude envelope
        if np.std(amplitude_envelope) > 0.1 * np.mean(amplitude_envelope):
            return "AM"
        
        # Check for Frequency Modulation (FM): Look for variations in instantaneous frequency
        if np.std(instantaneous_frequency) > 0.1 * np.mean(np.abs(instantaneous_frequency)):
            return "FM"
        
        # Check for Phase Modulation (PM): Look for variations in phase
        if np.std(np.diff(instantaneous_phase)) > 0.1 * np.mean(np.abs(np.diff(instantaneous_phase))):
            return "PM"
        
        # Check for Quadrature Amplitude Modulation (QAM): Look for significant in-phase and quadrature components
        in_phase = np.real(analytic_signal)
        quadrature = np.imag(analytic_signal)
        if np.std(in_phase) > 0.1 * np.mean(np.abs(in_phase)) and np.std(quadrature) > 0.1 * np.mean(np.abs(quadrature)):
            return "QAM"
        
        # If no clear modulation type is detected, return "Unknown"
        return "Unknown"


    """
    Picos Espectrales
    """
    def detect_peaks(self):
        """
        Detect peaks in the signal that are greater than the specified threshold.

        Returns:
        np.array: Indices of the peaks in the signal that are above the threshold.
        """
        # Find peaks in the signal that are above the threshold
        peaks, _ = signal.find_peaks(self.linear_mg, height=self.threshold)
        return peaks
    
    """
    Analisis de ocupación
    """
    def determine_bandwidth_utilization(fft_signal, sample_rate, bandwidth_min, bandwidth_max):
        """
        Determines how much of a given bandwidth is utilized based on the FFT signal.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal.
        - sample_rate: int, the sampling rate of the original time-domain signal (in Hz).
        - bandwidth_min: float, the lower bound of the bandwidth of interest (in Hz).
        - bandwidth_max: float, the upper bound of the bandwidth of interest (in Hz).

        Returns:
        - utilization_fraction: float, the fraction of total power used in the given bandwidth.
        """
        # Length of the FFT signal
        N = len(fft_signal)
        
        # Compute the magnitude (amplitude) of the FFT signal
        amplitude = np.abs(fft_signal[:N // 2])  # Only positive frequencies
        xf = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]  # Corresponding frequencies
        
        # Compute the total power of the FFT signal (power is amplitude squared)
        total_power = np.sum(amplitude**2)
        
        # Find indices that correspond to the given bandwidth range
        indices_in_band = np.where((xf >= bandwidth_min) & (xf <= bandwidth_max))[0]
        
        # Compute the power within the specified bandwidth
        power_in_band = np.sum(amplitude[indices_in_band]**2)
        
        # Calculate the fraction of power utilized in the given bandwidth
        utilization_fraction = power_in_band / total_power if total_power != 0 else 0
        
        return utilization_fraction
    
    """
    Crest factor
    """
    def calculate_crest_factor(fft_signal):
        """
        Calculates the crest factor of an FFT-transformed signal.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal.

        Returns:
        - crest_factor: float, the crest factor of the signal.
        """
        # Compute the magnitude (amplitude) of the FFT signal
        amplitude = np.abs(fft_signal)
        
        # Compute the peak amplitude
        peak_amplitude = np.max(amplitude)
        
        # Compute the RMS (Root Mean Square) amplitude
        rms_amplitude = np.sqrt(np.mean(amplitude**2))
        
        # Compute the crest factor (Peak / RMS)
        crest_factor = peak_amplitude / rms_amplitude if rms_amplitude != 0 else np.inf
        
        return crest_factor
    
    """
    PRF
    """
    def calculate_prf(fft_signal, sample_rate, low_freq_limit=1, high_freq_limit=1000):
        """
        Calculates the Pulse Repetition Frequency (PRF) of an FFT-transformed signal.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal.
        - sample_rate: int, the sample rate of the original time-domain signal (in Hz).
        - low_freq_limit: float, the lower bound of frequency (in Hz) to search for PRF (default=1 Hz).
        - high_freq_limit: float, the upper bound of frequency (in Hz) to search for PRF (default=1000 Hz).

        Returns:
        - prf: float, the estimated Pulse Repetition Frequency (PRF) in Hz.
        """
        # Length of the FFT signal
        N = len(fft_signal)
        
        # Compute the magnitude (amplitude) of the FFT signal
        amplitude = np.abs(fft_signal[:N // 2])  # Only consider positive frequencies
        xf = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]  # Corresponding frequencies
        
        # Filter out frequencies outside the PRF range of interest (low_freq_limit to high_freq_limit)
        freq_range_mask = (xf >= low_freq_limit) & (xf <= high_freq_limit)
        
        # Limit amplitude and frequency to the range of interest
        filtered_amplitude = amplitude[freq_range_mask]
        filtered_xf = xf[freq_range_mask]
        
        # Detect peaks in the filtered amplitude spectrum
        peaks, _ = find_peaks(filtered_amplitude)
        
        # If no peaks are found, return None
        if len(peaks) == 0:
            return None
        
        # The PRF is assumed to be the frequency of the highest peak within the selected range
        dominant_peak_idx = np.argmax(filtered_amplitude[peaks])
        prf = filtered_xf[peaks[dominant_peak_idx]]
        
        return prf

    """
    Interference
    """
    def evaluate_interference_by_channel(fft_signal, sample_rate, channel_centers, channel_bandwidth):
        """
        Evaluates interference for each channel in an FFT-transformed signal.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal (complex-valued).
        - sample_rate: int, the sample rate of the original time-domain signal (in Hz).
        - channel_centers: array-like, list of center frequencies for each channel (in Hz).
        - channel_bandwidth: float, bandwidth of each channel (in Hz).

        Returns:
        - channel_interference: list of tuples, where each tuple contains (center frequency, interference power).
        """
        # Length of the FFT signal
        N = len(fft_signal)
        
        # Compute the magnitude (amplitude) of the FFT signal
        amplitude = np.abs(fft_signal[:N // 2])  # Only consider positive frequencies
        xf = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]  # Corresponding frequencies
        
        # List to store interference power for each channel
        channel_interference = []
        
        # Loop through each channel center frequency
        for center_freq in channel_centers:
            # Define the frequency range for the current channel
            lower_bound = center_freq - channel_bandwidth / 2
            upper_bound = center_freq + channel_bandwidth / 2
            
            # Find the indices that correspond to the frequencies within the channel
            indices_in_channel = np.where((xf >= lower_bound) & (xf <= upper_bound))[0]
            
            # Calculate the total power (amplitude squared) within the channel
            power_in_channel = np.sum(amplitude[indices_in_channel]**2)
            
            # Append the result (center frequency, interference power) to the list
            channel_interference.append((center_freq, power_in_channel))
        
        return channel_interference
    
    """
    Frequency drift
    """
    def detect_main_frequency_changes_from_fft(fft_signal, sample_rate, nperseg=256, noverlap=None, window='hann'):
        """
        Detects changes in the main frequency of a signal across time using STFT on an FFT-transformed signal.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal.
        - sample_rate: int, the sample rate of the original time-domain signal (in Hz).
        - nperseg: int, length of each segment for the STFT (default=256).
        - noverlap: int, number of points to overlap between segments (default=None, meaning nperseg // 2).
        - window: str or tuple, desired window to use (default is 'hann').

        Returns:
        - time_bins: array, the time points corresponding to the main frequency changes.
        - main_frequencies: array, the main frequency at each time bin.
        """
        # Compute the Short-Time Fourier Transform (STFT)
        f, t, Zxx = stft(fft_signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window=window)
        
        # Get the magnitude of the STFT result (to find the main frequency)
        magnitude = np.abs(Zxx)
        
        # Detect the main frequency at each time bin (find the frequency with the highest magnitude)
        main_frequencies = f[np.argmax(magnitude, axis=0)]
        
        return t, main_frequencies
        

    """
    Tiempo ocupación
    """
    def detect_signal_presence_in_spectrum(fft_signal, sample_rate, threshold=0.01, nperseg=256, noverlap=None, window='hann'):
        """
        Detects how much time the signal is present in the spectrum using STFT.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal.
        - sample_rate: int, the sample rate of the original time-domain signal (in Hz).
        - threshold: float, amplitude threshold to detect signal presence (default=0.01).
        - nperseg: int, length of each segment for the STFT (default=256).
        - noverlap: int, number of points to overlap between segments (default=None, meaning nperseg // 2).
        - window: str or tuple, desired window to use (default is 'hann').

        Returns:
        - active_time: float, total time the signal is present (in seconds).
        - total_time: float, total duration of the signal (in seconds).
        - presence_fraction: float, fraction of time the signal is present in the spectrum.
        """
        # Compute the Short-Time Fourier Transform (STFT)
        f, t, Zxx = stft(fft_signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window=window)
        
        # Compute the magnitude of the STFT (to detect presence)
        magnitude = np.abs(Zxx)
        
        # Detect signal presence by checking if the max amplitude in each time window exceeds the threshold
        signal_present = np.any(magnitude > threshold, axis=0)
        
        # Calculate the duration for which the signal is present
        time_bin_duration = t[1] - t[0]  # Time difference between two consecutive time bins
        active_time = np.sum(signal_present) * time_bin_duration  # Total active time
        
        # Total duration of the signal
        total_time = t[-1] - t[0]
        
        # Fraction of the time the signal is present
        presence_fraction = active_time / total_time if total_time > 0 else 0
        
        return active_time, total_time, presence_fraction
    
    """
    Espectro temporal
    """

    def waterfall_display(fft_signal, sample_rate, nperseg=256, noverlap=None, window='hann', cmap='viridis'):
        """
        Generates a waterfall display of an FFT signal using STFT.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal.
        - sample_rate: int, the sample rate of the original time-domain signal (in Hz).
        - nperseg: int, length of each segment for the STFT (default=256).
        - noverlap: int, number of points to overlap between segments (default=None, meaning nperseg // 2).
        - window: str or tuple, desired window to use (default is 'hann').
        - cmap: str, the colormap to use for the display (default is 'viridis').

        Returns:
        - None, displays the waterfall plot.
        """
        # Compute the Short-Time Fourier Transform (STFT)
        f, t, Zxx = stft(fft_signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window=window)
        
        # Compute the magnitude of the STFT
        magnitude = np.abs(Zxx)
        
        # Plot the waterfall display using a 2D colormap
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, magnitude, shading='gouraud', cmap=cmap)
        plt.title('Waterfall Display of FFT Signal')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Magnitude')
        plt.show()

    def calculate_power_in_bandwidth(fft_signal, sample_rate, bandwidth_min, bandwidth_max):
        """
        Calculates the power of an FFT signal within a specific bandwidth channel.

        Parameters:
        - fft_signal: array-like, the FFT-transformed signal (complex-valued).
        - sample_rate: int, the sample rate of the original time-domain signal (in Hz).
        - bandwidth_min: float, the lower bound of the bandwidth (in Hz).
        - bandwidth_max: float, the upper bound of the bandwidth (in Hz).

        Returns:
        - power_in_band: float, the total power within the specified bandwidth.
        """
        # Length of the FFT signal
        N = len(fft_signal)
        
        # Compute the magnitude (amplitude) of the FFT signal
        amplitude = np.abs(fft_signal[:N // 2])  # Only positive frequencies
        xf = np.fft.fftfreq(N, 1 / sample_rate)[:N // 2]  # Corresponding positive frequencies
        
        # Find the indices corresponding to the frequency range within the specified bandwidth
        indices_in_band = np.where((xf >= bandwidth_min) & (xf <= bandwidth_max))[0]
        
        # Calculate the power (amplitude squared) within the specified bandwidth
        power_in_band = np.sum(amplitude[indices_in_band]**2)
        
        return power_in_band


    def detect_valleys(self):
        """
        Detect valleys in the signal that are below the specified threshold.

        Returns:
        np.array: Indices of the valleys in the signal that are below the threshold.
        """
        # Find valleys in the signal that are below the threshold
        valleys, _ = signal.find_peaks(-self.linear_mg, height=-self.threshold)
        return valleys

    def carry_signal(self):
        """
        Extract the signal from the dataset and find the index of the maximum value.
        
        Returns:
        int: Index of the maximum absolute value in the signal.
        """
        # Extract the 'signal' column from the DataFrame and convert it to a NumPy array
        self.signal = self.linear_mg.to_numpy()
        # Find the index of the maximum absolute value in the signal
        self.max_value = np.argmax(np.abs(self.signal))
        return self.max_value

    def suavizar(self, window_length=11, polyorder=2):
        """
        Aplica un filtro de suavizado usando el filtro de Savitzky-Golay.
        
        Parameters:
        window_length (int): Longitud de la ventana del filtro.
        polyorder (int): Orden del polinomio que se ajustará.
        """
        self.amplitud_suavizada = savgol_filter(self.linear_mg, window_length, polyorder)
        return self.amplitud_suavizada
    
    def interpolar(self):
        interpolador = interp1d(self.frecuency, self.linear_mg, kind='cubic')
        nueva_frecuencia = np.linspace(min(self.frecuency), max(self.frecuency), len(self.linear_mg))
        self.amplitud_interpolada = interpolador(nueva_frecuencia)
        return self.amplitud_interpolada

    def plot(self):
        """
        Plot the signal data.
        """
        # Plot the data (this will plot all columns in the DataFrame, not just 'signal')
        plt.plot(self.frecuency, self.linear_mg)
        plt.grid()
        plt.title('Signal Plot')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
    
    def plot_2(self):
        """
        Plot the signal data.
        """
        # Plot the data (this will plot all columns in the DataFrame, not just 'signal')
        plt.plot(self.frecuency, self.amplitud_suavizada, color='r', label='Suavizada')
        plt.plot(self.frecuency, self.linear_mg, color='b', linestyle='--', label='Original')
        plt.grid()
        plt.title('Signal Plot')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.legend()  # Esto automáticamente toma las etiquetas de los parámetros 'label'
        plt.show()

    def plot_3(self):
        """
        Plot the signal data.
        """
        # Plot the data (this will plot all columns in the DataFrame, not just 'signal')
        plt.plot(self.frecuency, self.amplitud_suavizada, color='r', label='Suavizada')
        plt.plot(self.frecuency, self.linear_mg, color='b', linestyle='--', label='Original')
        plt.plot(self.frecuency, self.amplitud_interpolada, color='g', label='interpolada')
        plt.grid()
        plt.title('Signal Plot')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.legend()  # Esto automáticamente toma las etiquetas de los parámetros 'label'
        plt.show()
    
    def plot_suavisado(self):
        """
        Plot the signal data.
        """
        # Plot the data (this will plot all columns in the DataFrame, not just 'signal')
        plt.plot(self.frecuency, self.amplitud_suavizada)
        plt.grid()
        plt.title('Signal Plot')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def power():
        pass


    def autocorrelation(self):
        """
        Compute and plot the autocorrelation of the signal.
        """
        N = len(self.data['signal'])  # Length of the signal
        autocorr = np.correlate(self.data['signal'], self.data['signal'], mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Keep only the second half

        # Plot the autocorrelation result
        plt.plot(autocorr)
        plt.title('Autocorrelation of Signal')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()

    def cross_correlation(self, data2):
        """
        Compute and plot the cross-correlation between the signal and another signal.

        Parameters:
        data2 (pd.Series or np.array): Second signal for cross-correlation.
        """
        # Convert data2 to a NumPy array if it is not already
        if isinstance(data2, pd.Series):
            data2 = data2.to_numpy()
        
        # Perform cross-correlation
        cross_corr = np.correlate(self.data['signal'], data2, mode='full')

        # Plot the cross-correlation result
        plt.plot(cross_corr)
        plt.title('Cross-Correlation of Signals')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.show()


if __name__ == "__main__":
   signal_analitics = SignalAnalytics("DATA_1.csv", 0.1)
   signal_analitics.suavizar(window_length=11, polyorder=2)
   signal_analitics.interpolar()
   signal_analitics.plot_3()
   print("Frecuencia central: ") # Hz 
   signal_analitics.frecuenciaCentral()
   print("Ancho de banda: ") # Hz 
   signal_analitics.anchoDeBanda(threshold_db=-3)
   print("Potencia de la señal: ") # watts
   signal_analitics.potencia()
   print("SNR ") # dB 
   signal_analitics.calcular_snr_sin_ruido()
   print("Frecuencias de spurias")  # Hz
   signal_analitics.frecuenciasSpuria()
   