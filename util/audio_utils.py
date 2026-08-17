import numpy as np
import torchaudio
import torch
import soundfile as sf
from typing import Tuple, Optional

def load_audio_pi(audio_path):
    # Load the audio data and sample rate
    data, samplerate = sf.read(audio_path, dtype='float32')

    # Convert to PyTorch tensor and reshape to (channels, time)
    waveform = torch.from_numpy(data).T

    # Ensure mono audio has a channel dimension: (1, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    return waveform, samplerate


def read_audio_to_lpcm_bytes(
    audio_path: str,
    sample_rate: Optional[int] = None,
    mono: bool = True,
    bit_depth: int = 16
) -> Tuple[bytes, int]:
    """
    Legge un file audio e lo converte in array di byte LPCM.
    
    Args:
        audio_path (str): Percorso del file audio
        sample_rate (int, optional): Frequenza di campionamento desiderata.
                                     Se None, usa la frequenza del file
        mono (bool): Se True, converte a mono; se False, mantiene i canali
        bit_depth (int): Profondità di bit (default: 16)
    
    Returns:
        Tuple[bytes, int]: (array di byte LPCM, sample rate usato)
    
    Esempio:
        >>> lpcm_bytes, sr = read_audio_to_lpcm_bytes("audio.wav", sample_rate=16000)
        >>> print(f"Sample rate: {sr}, Dimensione in byte: {len(lpcm_bytes)}")
    """
    # Carica il file audio
    waveform, original_sr = torchaudio.load(audio_path)
    # waveform, original_sr = load_audio_pi(audio_path)
    
    # Determina il sample rate da usare
    target_sr = sample_rate if sample_rate is not None else original_sr
    
    # Resample se necessario
    if target_sr != original_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)
    
    # Converti a mono se richiesto
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalizza e converti a PCM
    if bit_depth == 16:
        # Scala da [-1, 1] a [-32768, 32767]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).to(torch.int16)
    elif bit_depth == 24:
        # Scala da [-1, 1] a [-8388608, 8388607]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 8388607).to(torch.int32)
    elif bit_depth == 32:
        # Scala da [-1, 1] a [-2147483648, 2147483647]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 2147483647).to(torch.int32)
    else:
        raise ValueError(f"Bit depth non supportato: {bit_depth}")
    
    # Appiattisci e converti a numpy
    waveform_np = waveform.numpy().flatten()
    
    # Converti a byte
    lpcm_bytes = waveform_np.astype(f'int{bit_depth}').tobytes()
    
    return lpcm_bytes, target_sr


def read_audio_to_lpcm_array(
    audio_path: str,
    sample_rate: Optional[int] = None,
    mono: bool = True,
    bit_depth: int = 16
) -> Tuple[np.ndarray, int]:
    """
    Legge un file audio e lo converte in array numpy LPCM.
    
    Args:
        audio_path (str): Percorso del file audio
        sample_rate (int, optional): Frequenza di campionamento desiderata.
                                     Se None, usa la frequenza del file
        mono (bool): Se True, converte a mono; se False, mantiene i canali
        bit_depth (int): Profondità di bit (default: 16)
    
    Returns:
        Tuple[np.ndarray, int]: (array LPCM, sample rate usato)
    
    Esempio:
        >>> lpcm_array, sr = read_audio_to_lpcm_array("audio.wav", sample_rate=16000)
        >>> print(f"Shape: {lpcm_array.shape}, dtype: {lpcm_array.dtype}")
    """
    # Carica il file audio
    waveform, original_sr = torchaudio.load(audio_path)
    
    # Determina il sample rate da usare
    target_sr = sample_rate if sample_rate is not None else original_sr
    
    # Resample se necessario
    if target_sr != original_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)
    
    # Converti a mono se richiesto
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalizza e converti a PCM
    if bit_depth == 16:
        # Scala da [-1, 1] a [-32768, 32767]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).to(torch.int16)
    elif bit_depth == 24:
        # Scala da [-1, 1] a [-8388608, 8388607]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 8388607).to(torch.int32)
    elif bit_depth == 32:
        # Scala da [-1, 1] a [-2147483648, 2147483647]
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 2147483647).to(torch.int32)
    else:
        raise ValueError(f"Bit depth non supportato: {bit_depth}")
    
    # Appiattisci e converti a numpy
    lpcm_array = waveform.numpy().flatten().astype(f'int{bit_depth}')
    
    return lpcm_array, target_sr


def save_lpcm_bytes_to_file(
    lpcm_bytes: bytes,
    output_path: str,
    sample_rate: int,
    bit_depth: int = 16,
    num_channels: int = 1
) -> None:
    """
    Salva byte LPCM in un file audio WAV.
    
    Args:
        lpcm_bytes (bytes): Array di byte LPCM
        output_path (str): Percorso del file di output
        sample_rate (int): Frequenza di campionamento
        bit_depth (int): Profondità di bit (default: 16)
        num_channels (int): Numero di canali (default: 1 - mono)
    
    Esempio:
        >>> lpcm_bytes, sr = read_audio_to_lpcm_bytes("input.wav")
        >>> save_lpcm_bytes_to_file(lpcm_bytes, "output.wav", sr)
    """
    # Converti byte a numpy array
    lpcm_array = np.frombuffer(lpcm_bytes, dtype=f'int{bit_depth}')
    
    # Reshape in (canali, campioni)
    lpcm_array = lpcm_array.reshape(num_channels, -1)
    
    # Converti a float32 normalizzato in [-1, 1]
    if bit_depth == 16:
        waveform = lpcm_array.astype(np.float32) / 32767.0
    elif bit_depth == 24:
        waveform = lpcm_array.astype(np.float32) / 8388607.0
    elif bit_depth == 32:
        waveform = lpcm_array.astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Bit depth non supportato: {bit_depth}")
    
    # Converti a torch tensor e salva
    waveform_tensor = torch.from_numpy(waveform)
    torchaudio.save(output_path, waveform_tensor, sample_rate)
