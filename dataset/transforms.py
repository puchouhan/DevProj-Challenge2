import numpy as np
import torch
import librosa
import random


# Composes several transforms together.
class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


class RandomNoise:
    def __init__(self, min_noise=0.0, max_noise=0.05): #0.002, 0.01
        super(RandomNoise, self).__init__()
        
        self.min_noise = min_noise
        self.max_noise = max_noise
        
    def addNoise(self, wave):
        noise_val = random.uniform(self.min_noise, self.max_noise)
        noise = torch.from_numpy(np.random.normal(0, noise_val, wave.shape[0]))
        noisy_wave = wave + noise
        
        return noisy_wave
    
    def __call__(self, x):
        return self.addNoise(x)


class RandomScale:

    def __init__(self, max_scale: float = 1.25):
        super(RandomScale, self).__init__()

        self.max_scale = max_scale

    @staticmethod
    def random_scale(max_scale: float, signal: torch.Tensor) -> torch.Tensor:
        scaling = np.power(max_scale, np.random.uniform(-1, 1)) #between 1.25**(-1) and 1.25**(1)
        output_size = int(signal.shape[-1] * scaling)
        ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
        
        # ref1 is of size output_size
        ref1 = ref.clone().type(torch.int64)
        ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
        
        r = ref - ref1.type(ref.type())
        
        scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

        return scaled_signal

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_scale(self.max_scale, x)


class RandomCrop:

    def __init__(self, out_len: int = 44100, train: bool = True):
        super(RandomCrop, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_crop(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, signal.shape[-1] - self.out_len)
        else:
            left = int(round(0.5 * (signal.shape[-1] - self.out_len)))

        orig_std = signal.float().std() * 0.5
        output = signal[..., left:left + self.out_len]

        out_std = output.float().std()
        if out_std < orig_std:
            output = signal[..., :self.out_len]

        new_out_std = output.float().std()
        if orig_std > new_out_std > out_std:
            output = signal[..., -self.out_len:]

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_crop(x) if x.shape[-1] > self.out_len else x


class RandomPadding:

    def __init__(self, out_len: int = 88200, train: bool = True):
        super(RandomPadding, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:
        
        if self.train:
            left = np.random.randint(0, self.out_len - signal.shape[-1])
        else:
            left = int(round(0.5 * (self.out_len - signal.shape[-1])))

        right = self.out_len - (left + signal.shape[-1])

        pad_value_left = signal[..., 0].float().mean().to(signal.dtype)
        pad_value_right = signal[..., -1].float().mean().to(signal.dtype)
        output = torch.cat((
            torch.zeros(signal.shape[:-1] + (left,), dtype=signal.dtype, device=signal.device).fill_(pad_value_left),
            signal,
            torch.zeros(signal.shape[:-1] + (right,), dtype=signal.dtype, device=signal.device).fill_(pad_value_right)
        ), dim=-1)

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x) if x.shape[-1] < self.out_len else x


class FrequencyMask():
    def __init__(self, max_width, numbers):
        super(FrequencyMask, self).__init__()

        self.max_width = max_width
        self.numbers = numbers

    def addFreqMask(self, wave):
        #print(wave.shape)
        for _ in range(self.numbers):
            #choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[1] - mask_len) #start of the mask
            end = start + mask_len
            wave[:, start:end, : ] = 0

        return wave

    def __call__(self, wave):
        return self.addFreqMask(wave)


class TimeMask():
    def __init__(self, max_width, numbers):
        super(TimeMask, self).__init__()

        self.max_width = max_width
        self.numbers = numbers


    def addTimeMask(self, wave):

        for _ in range(self.numbers):
            #choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[2] - mask_len) #start of the mask
            end = start + mask_len
            wave[ : , : , start:end] = 0

        return wave

    def __call__(self, wave):
        return self.addTimeMask(wave)


class RandomNoise:
    """
    Fügt zufälliges Gaußsches Rauschen zum Audiosignal hinzu.
    """

    def __init__(self, min_noise=0.001, max_noise=0.01):
        """
        Initialisiert die RandomNoise-Transformation.

        Args:
            min_noise (float): Minimale Standardabweichung des Rauschens
            max_noise (float): Maximale Standardabweichung des Rauschens
        """
        super(RandomNoise, self).__init__()

        self.min_noise = min_noise
        self.max_noise = max_noise

    def addNoise(self, wave):
        """
        Fügt weißes Rauschen mit zufälliger Stärke zum Signal hinzu.
        """
        noise_val = random.uniform(self.min_noise, self.max_noise)
        # Stellen Sie sicher, dass die Form des Rauschens mit der Eingabe übereinstimmt
        noise = torch.from_numpy(np.random.normal(0, noise_val, wave.shape))
        noisy_wave = wave + noise

        return noisy_wave

    def __call__(self, x):
        return self.addNoise(x)


class RandomVolume:
    """
    Ändert die Lautstärke des Audiosignals zufällig.
    """

    def __init__(self, min_gain=0.5, max_gain=1.5):
        """
        Initialisiert die RandomVolume-Transformation.

        Args:
            min_gain (float): Minimaler Verstärkungsfaktor
            max_gain (float): Maximaler Verstärkungsfaktor
        """
        super(RandomVolume, self).__init__()

        self.min_gain = min_gain
        self.max_gain = max_gain

    def adjust_volume(self, wave):
        """
        Ändert die Lautstärke durch Multiplikation mit einem zufälligen Faktor.
        """
        gain = random.uniform(self.min_gain, self.max_gain)
        adjusted_wave = wave * gain

        # Clipping vermeiden bei zu hoher Verstärkung
        if gain > 1.0:
            max_val = torch.max(torch.abs(adjusted_wave))
            if max_val > 32767.0:  # Typischer Maximalwert für 16-bit Audio
                adjusted_wave = adjusted_wave * (32767.0 / max_val)

        return adjusted_wave

    def __call__(self, x):
        return self.adjust_volume(x)


class RandomTimeShift:
    """
    Verschiebt das Audiosignal zufällig in der Zeit.
    """

    def __init__(self, max_shift_sec=1.0, sr=44100):
        """
        Initialisiert die RandomTimeShift-Transformation.

        Args:
            max_shift_sec (float): Maximale Verschiebung in Sekunden
            sr (int): Sampling-Rate des Audiosignals
        """
        super(RandomTimeShift, self).__init__()

        self.max_shift_samples = int(max_shift_sec * sr)

    def shift_time(self, wave):
        """
        Verschiebt das Signal zufällig nach links oder rechts und füllt die
        entstehende Lücke mit Stille auf.
        """
        # Überprüfen der Form des Eingangs
        if len(wave.shape) == 1:
            # Eindimensionales Signal (nur Samples)
            shift = random.randint(-self.max_shift_samples, self.max_shift_samples)

            # Kein Shift notwendig
            if shift == 0:
                return wave

            # Verschiebung nach rechts
            if shift > 0:
                shifted_wave = torch.cat([
                    torch.zeros(shift, device=wave.device, dtype=wave.dtype),
                    wave[:-shift] if shift < wave.shape[0] else torch.tensor([], device=wave.device, dtype=wave.dtype)
                ])
            # Verschiebung nach links
            else:
                shift = abs(shift)
                shifted_wave = torch.cat([
                    wave[shift:],
                    torch.zeros(shift, device=wave.device, dtype=wave.dtype)
                ])

            # Sicherstellen, dass die Ausgabe die gleiche Länge hat wie die Eingabe
            if shifted_wave.shape[0] > wave.shape[0]:
                shifted_wave = shifted_wave[:wave.shape[0]]
            elif shifted_wave.shape[0] < wave.shape[0]:
                shifted_wave = torch.cat([
                    shifted_wave,
                    torch.zeros(wave.shape[0] - shifted_wave.shape[0], device=wave.device, dtype=wave.dtype)
                ])

            return shifted_wave

        elif len(wave.shape) == 2:
            # Zweidimensionales Signal (Kanäle, Samples)
            n_channels, n_samples = wave.shape
            shift = random.randint(-self.max_shift_samples, self.max_shift_samples)

            # Kein Shift notwendig
            if shift == 0:
                return wave

            # Ausgabe-Tensor mit der gleichen Form wie Eingabe erstellen
            shifted_wave = torch.zeros_like(wave)

            # Verschiebung nach rechts
            if shift > 0:
                if shift < n_samples:
                    shifted_wave[:, shift:] = wave[:, :n_samples - shift]
            # Verschiebung nach links
            else:
                shift = abs(shift)
                if shift < n_samples:
                    shifted_wave[:, :n_samples - shift] = wave[:, shift:]

            return shifted_wave

        else:
            raise ValueError(f"Unerwartete Dimensionalität des Signals: {wave.shape}")

    def __call__(self, x):
        return self.shift_time(x)


class RandomPitch:
    """
    Führt eine zufällige Tonhöhenverschiebung (Pitch Shift) auf ein Audiosignal durch.
    """

    def __init__(self, max_steps=4, min_steps=-4):
        """
        Initialisiert die RandomPitch-Transformation.

        Args:
            max_steps (int): Maximale Anzahl der Halbtonschritte für die Erhöhung der Tonhöhe.
            min_steps (int): Minimale Anzahl der Halbtonschritte für die Senkung der Tonhöhe.
        """
        super(RandomPitch, self).__init__()

        self.max_steps = max_steps
        self.min_steps = min_steps

    def shift_pitch(self, audio):
        """
        Führt die zufällige Tonhöhenverschiebung durch.

        Args:
            audio (torch.Tensor): Das Eingabe-Audiosignal.

        Returns:
            torch.Tensor: Das transformierte Audiosignal.
        """
        # Zufällige Anzahl der Halbtonschritte für die Verschiebung
        n_steps = np.random.uniform(self.min_steps, self.max_steps)

        # Konvertierung zu NumPy, falls es ein Torch-Tensor ist
        is_torch = isinstance(audio, torch.Tensor)
        if is_torch:
            # Dimension speichern
            original_shape = audio.shape
            original_device = audio.device
            original_dtype = audio.dtype

            # Zu NumPy konvertieren
            if len(original_shape) == 2:  # [channels, samples]
                audio_np = audio.cpu().numpy()
                # Transponieren für librosa (samples, channels)
                if original_shape[0] <= 4:  # Vermutlich Kanäle zuerst
                    audio_np = audio_np.T
                    transposed = True
                else:
                    transposed = False
            else:
                audio_np = audio.cpu().numpy()
                transposed = False
        else:
            audio_np = audio
            transposed = False
            original_shape = audio.shape

        # Pitch Shift anwenden
        if audio_np.ndim == 1 or (audio_np.ndim == 2 and audio_np.shape[1] <= 4):
            # Für eindimensionales Audio oder mehrkanaliges Audio
            shifted = librosa.effects.pitch_shift(
                y=audio_np,
                sr=44100,
                n_steps=n_steps
            )
        else:
            # Für mehrdimensionales Audio (mehrere Kanäle)
            shifted = np.apply_along_axis(
                lambda x: librosa.effects.pitch_shift(x, sr=44100, n_steps=n_steps),
                axis=0,
                arr=audio_np
            )

        # Zurück zu Torch-Tensor, falls das Original ein Torch-Tensor war
        if is_torch:
            # Wenn es transponiert wurde, rücktransponieren
            if transposed:
                shifted = shifted.T

            # Zurück zum Tensor und ursprüngliches Gerät/Datentyp
            shifted_tensor = torch.from_numpy(shifted).to(device=original_device, dtype=original_dtype)

            # Stellen Sie sicher, dass die Form übereinstimmt
            if shifted_tensor.shape != original_shape:
                # Falls die Länge unterschiedlich ist, anpassen
                if len(original_shape) == 1:
                    if shifted_tensor.shape[0] < original_shape[0]:
                        padding = torch.zeros(original_shape[0] - shifted_tensor.shape[0],
                                              device=original_device, dtype=original_dtype)
                        shifted_tensor = torch.cat([shifted_tensor, padding])
                    else:
                        shifted_tensor = shifted_tensor[:original_shape[0]]
                else:  # 2D
                    if shifted_tensor.shape[1] < original_shape[1]:
                        padding = torch.zeros(original_shape[0], original_shape[1] - shifted_tensor.shape[1],
                                              device=original_device, dtype=original_dtype)
                        shifted_tensor = torch.cat([shifted_tensor, padding], dim=1)
                    else:
                        shifted_tensor = shifted_tensor[:, :original_shape[1]]

            return shifted_tensor

        return shifted

    def __call__(self, x):
        return self.shift_pitch(x)


