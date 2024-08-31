import os
import json
import base64
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyfftw
import tqdm
import soxr
import pandas as pd
import ruptures as rpt
from IPython.display import display


from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.inference import Spect2Frames, load_model


def signal2spect(signal, sr, device):
        if signal.ndim == 2:
            signal = signal.mean(1)
        elif signal.ndim != 1:
            raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device=device)
        spect = LogMelSpect(device=device)
        return spect(signal)


def analyze_beat(audio_path, penalty=1000):
    num_windows = 21

    fps = 50
    signal, sr = load_audio(audio_path)
    sampling_per_frame = sr//fps
    sampling_per_window = sampling_per_frame/num_windows
    window_per_sec = fps * num_windows

    model_names = ['beat_this-final0.ckpt', 'beat_this-final1.ckpt', 'beat_this-final2.ckpt']

    num_ensemble = len(model_names)
    beat_activations = []
    downbeat_activations = []
    window_spect = []
    signal, sr = load_audio(audio_path)

    for i in range(num_windows):
        print(f'STFT {i+1}/{num_windows}...', end="\r")
        window_signal = np.insert(np.zeros([sampling_per_frame, 2]), int(np.round(sampling_per_window*i)), signal, axis=0)
        window_spect.append(signal2spect(window_signal, sr, 'cuda'))
    print('\n')
        
    for i, model_name in enumerate(model_names):
        beat_activation = []
        downbeat_activation = []
        model = Spect2Frames(model_name, device='cuda', float16=True)
        for ii in range(num_windows):
            print(f'Model {i}: Window {ii}', end="\r")
            beat_pred, downbeat_pred = model(window_spect[ii])
            beat_activation.append(torch.sigmoid(beat_pred).detach().cpu().numpy())
            downbeat_activation.append(torch.sigmoid(downbeat_pred).detach().cpu().numpy())
        beat_activations.append(np.stack(beat_activation))
        downbeat_activations.append(np.stack(downbeat_activation))
        print('\n')
    
    beat_activations = np.stack(beat_activations, axis=1)
    downbeat_activations = np.stack(downbeat_activations, axis=1)
    
    result = analyze_activation(beat_activations, num_windows, num_ensemble, sr, fps, penalty=penalty)
        
    return result

beat_wisdom = (b'(fftw-3.3.5 fftw_wisdom #x3c273403 #x192df114 #x4d08727c #xe98e9b9d\n  (fftw_rdft_nop_register 0 #x1040 #x1040 #x0 #x283dce8d #x3449a408 #x29007213 #xd983052a)\n  (fftw_rdft_nop_register 0 #x1040 #x1040 #x0 #xd0702255 #xe76a9ab3 #x5398ff07 #x60ee718e)\n  (fftw_codelet_hf_7 0 #x11bdd #x11bdd #x0 #xc12091e3 #xf10fcfa6 #xbb04e408 #xe316f430)\n  (fftw_rdft_vrank_geq1_register 0 #x1040 #x1040 #x0 #xe39ba597 #x320a5770 #xf81c9404 #x0cb8a00e)\n  (fftw_dft_vrank_geq1_register 0 #x1040 #x1040 #x0 #xe5385d79 #xc0e088b3 #x9c2b311b #x3bd0db8a)\n  (fftw_rdft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #x5ce75b97 #x0208d7c4 #x1b95c872 #xacb1493f)\n  (fftw_rdft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #x38439018 #xbb95c09b #x023074c3 #x1ec2890e)\n  (fftw_codelet_r2cf_9 0 #x11bdd #x11bdd #x0 #x31f540d5 #x74206f9f #xdcff2fdd #x4cfd7ee1)\n  (fftw_rdft_nop_register 0 #x11bdd #x11bdd #x0 #xfbf6e89d #x095c9c97 #x14434e4a #x5c3bd15f)\n  (fftw_codelet_r2cf_2 2 #x1040 #x1040 #x0 #x6e835f11 #x4e2c3842 #x5c96cc7e #xa0fb0461)\n  (fftw_rdft2_nop_register 0 #x11bdd #x11bdd #x0 #x457bb0a2 #xbec2f8c7 #x2fa67fb9 #x7a401041)\n  (fftw_dft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #xa4a72150 #x4a6f9c4a #xb7c9ace4 #x769b8aca)\n  (fftw_codelet_r2cf_15 0 #x11bdd #x11bdd #x0 #x8a6a8c0f #x6676c8f6 #x0075c633 #xa922dc67)\n  (fftw_codelet_r2cf_15 0 #x1040 #x1040 #x0 #x5433655b #x5797c058 #xbf2cb1a3 #x659e36da)\n  (fftw_codelet_hf_25 0 #x11bdd #x11bdd #x0 #x36a21760 #x7ff2eddd #x2a422ca4 #xa093f1c5)\n  (fftw_codelet_hf_15 0 #x11bdd #x11bdd #x0 #xb86c9720 #xaa4730e5 #xa0d9dd66 #x0dd38bcd)\n  (fftw_codelet_t1fv_7_avx 0 #x11bdd #x11bdd #x0 #x919fe1c9 #xa901b999 #x2595fa6b #x394adc59)\n  (fftw_codelet_r2cf_7 0 #x11bdd #x11bdd #x0 #x9e4a85e7 #x6bfe07ab #x653a6e0d #x02afc4b3)\n  (fftw_codelet_hf_5 0 #x1040 #x1040 #x0 #x5133e3e3 #x89f79eb2 #x888d1466 #x5f83b353)\n  (fftw_rdft2_nop_register 0 #x1040 #x1040 #x0 #x457bb0a2 #xbec2f8c7 #x2fa67fb9 #x7a401041)\n  (fftw_codelet_r2cf_25 0 #x11bdd #x11bdd #x0 #x2223136b #xc64629e5 #xc703ae36 #x2067f0aa)\n  (fftw_codelet_r2cf_15 0 #x1040 #x1040 #x0 #x6b884fed #x410aca7c #xd48b6b21 #xe71d6d71)\n  (fftw_codelet_r2cf_15 0 #x1040 #x1040 #x0 #x8b018d61 #x7e97d4ca #xc0420d65 #x5dc0663a)\n  (fftw_dft_vrank_geq1_register 0 #x1040 #x1040 #x0 #xe2cdedb4 #x3090026f #xe8a00b9d #x486e4534)\n  (fftw_rdft_vrank_geq1_register 0 #x1040 #x1040 #x0 #xd969ff59 #x9825cda7 #xd3d7e731 #x061b6833)\n  (fftw_rdft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #xc06dd58b #xc2db362b #xda77827f #x19a3489c)\n  (fftw_codelet_hf_15 0 #x1040 #x1040 #x0 #xcf942b90 #x87e6f9b3 #xf4c152ac #xc67c5286)\n  (fftw_codelet_hf_7 0 #x1040 #x1040 #x0 #x53ecfbbc #xe77b68de #x005de680 #xb07ab398)\n  (fftw_dft_vrank_geq1_register 0 #x1040 #x1040 #x0 #xe76ea216 #xd7f0c8eb #xb19fef1d #xf5b2bab3)\n  (fftw_rdft_vrank_geq1_register 0 #x1040 #x1040 #x0 #x978b1995 #xd73e8fbf #x60d2ff09 #x61c6f2e3)\n  (fftw_codelet_hc2cfdftv_2_avx 0 #x1040 #x1040 #x0 #x6f92d285 #x214a1f4f #x1058ba3b #xaa66e0ed)\n  (fftw_rdft_nop_register 0 #x1040 #x1040 #x0 #xfbf6e89d #x095c9c97 #x14434e4a #x5c3bd15f)\n  (fftw_codelet_t3fv_20_avx 0 #x1040 #x1040 #x0 #x0b3f9c1b #xdb3f79da #x24fbaddd #x99868561)\n  (fftw_codelet_r2cf_5 0 #x1040 #x1040 #x0 #xc6a55261 #x11876998 #xba67a673 #x59ed0f0b)\n  (fftw_dft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #x6fe2ca45 #x06ac9e35 #x2ab80b22 #xc84803f1)\n  (fftw_codelet_r2cf_12 2 #x11bdd #x11bdd #x0 #x5eccafe1 #x5234a84a #x21d7237d #xf38e1495)\n  (fftw_codelet_r2cf_5 0 #x11bdd #x11bdd #x0 #xa7529567 #x89e1454d #xd5ef51cd #xf66cb539)\n  (fftw_codelet_hf_7 0 #x1040 #x1040 #x0 #x53b70f8c #x127a4c5b #x4ee7a4a3 #xa6367e56)\n  (fftw_codelet_n1fv_20_avx 0 #x11bdd #x11bdd #x0 #x892ec0f6 #xbaad569b #x791ab227 #x8140515b)\n  (fftw_codelet_hf_5 0 #x11bdd #x11bdd #x0 #xcf942b90 #x87e6f9b3 #xf4c152ac #xc67c5286)\n  (fftw_dft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #x9f7527a5 #xd773a5b4 #x65b5cd61 #x21a4d7b1)\n  (fftw_rdft_nop_register 0 #x11bdd #x11bdd #x0 #x283dce8d #x3449a408 #x29007213 #xd983052a)\n  (fftw_rdft_vrank_geq1_register 0 #x11bdd #x11bdd #x0 #xc44002bb #x5c6c5a61 #xf184d0c0 #x9829e10d)\n  (fftw_codelet_hc2cfdftv_12_avx 0 #x11bdd #x11bdd #x0 #x6f92d285 #x214a1f4f #x1058ba3b #xaa66e0ed)\n  (fftw_codelet_t2fv_5_avx 0 #x1040 #x1040 #x0 #x3e2bbc8f #xac718c41 #xb75d8c7e #x4f8307b9)\n  (fftw_codelet_hf_9 0 #x11bdd #x11bdd #x0 #x69bb57a9 #xd2e20204 #x476eb507 #x79664858)\n  (fftw_rdft2_rdft_register 0 #x1040 #x1040 #x0 #x824cf3e2 #x06f6241b #x318f2590 #x42adfad1)\n  (fftw_codelet_n2fv_14_avx 0 #x1040 #x1040 #x0 #x568b4775 #x7d3d1781 #x687ca263 #xf0c7f042)\n  (fftw_codelet_t3fv_25_avx 0 #x11bdd #x11bdd #x0 #xfe0ac0fc #xec234998 #xaa5007c0 #x593207f2)\n  (fftw_rdft_vrank_geq1_register 0 #x1040 #x1040 #x0 #x735f8869 #x38eb2d36 #x33388efc #x3aaf331b)\n  (fftw_codelet_r2cf_7 0 #x1040 #x1040 #x0 #x33f12c87 #xaac26329 #xc9752b89 #x7423f463)\n  (fftw_codelet_t1fv_15_avx 0 #x11bdd #x11bdd #x0 #xde964636 #x0316f0d6 #xa9501c20 #xc5fa0e17)\n  (fftw_codelet_r2cf_7 0 #x1040 #x1040 #x0 #x681d1d96 #x6757bbaf #xbbf92c4a #x4a951ff1)\n  (fftw_codelet_t1fv_15_avx 0 #x1040 #x1040 #x0 #x1cc32ed6 #xb0c561a0 #x72095a7f #x532e5f0a)\n  (fftw_codelet_t1fv_15_avx 0 #x1040 #x1040 #x0 #x6efe92a9 #x49f04836 #x8bafd0b6 #xa812c044)\n  (fftw_rdft_nop_register 0 #x11bdd #x11bdd #x0 #x114f3dc2 #xbf1f951b #x42acea82 #x123fd938)\n  (fftw_codelet_r2cf_7 0 #x11bdd #x11bdd #x0 #x5f8fd9ca #x7750fe69 #x33448554 #xca74a37a)\n  (fftw_codelet_r2cfII_2 2 #x1040 #x1040 #x0 #xef148d20 #xcab4a12a #xf07a145d #x4b3c02dd)\n  (fftw_codelet_hf_15 0 #x1040 #x1040 #x0 #x5db5cb62 #xf1ca75c4 #x4bbe6fb0 #xd3403d8e)\n  (fftw_rdft2_rdft_register 0 #x13c5 #x11bdd #x0 #x824cf3e2 #x06f6241b #x318f2590 #x42adfad1)\n  (fftw_codelet_r2cfII_12 2 #x11bdd #x11bdd #x0 #xbbf5bbba #xed8161ee #xa21938a4 #x538bb11c)\n)\n',
 b'(fftw-3.3.5 fftwf_wisdom #x706526c0 #x2f8b6c85 #x8cd1bb1a #x7c96e03d\n)\n',
 b'(fftw-3.3.5 fftwl_wisdom #x0821b5c7 #xa4c07d5a #x21b58211 #xebe513ab\n)\n')

pyfftw.import_wisdom(beat_wisdom)

def get_bpm_prior(integer_prior = 1.6, half_integer_prior = 1.4, primary_range = (80.0, 200.0), secondary_range = (60.0, 240.0), primary_prior = 1.2, secondary_prior = 1.1):
    prior = np.ones(10000)
    prior[0::10] *= integer_prior
    prior[5::10] *= half_integer_prior
    prior[int(secondary_range[0]*10):int(secondary_range[1]*10)] *= secondary_prior
    prior[int(primary_range[0]*10):int(primary_range[1]*10)] *= primary_prior/secondary_prior
    prior /= np.mean(prior)
    return prior

def dft_coeffs(x, N=None, maxk=None, step=1000):
    if N is None:
        N = len(x)
    if maxk is None:
        maxk = len(x)
        
    
    padded_activation = pyfftw.empty_aligned(N, dtype='float64')
    fft_result = pyfftw.empty_aligned(N//2 + 1, dtype='complex128')
    fft_object = pyfftw.FFTW(padded_activation, fft_result, flags=('FFTW_WISDOM_ONLY',))
    
    coeffs = []
    
    padded_activation[:] = pyfftw.zeros_aligned(N, dtype='float64')
    
    for i in range(-(len(x)//-step)):
        padded_activation[:] = pyfftw.zeros_aligned(N, dtype='float64')
        padded_activation[i*step:min((i+1)*step, len(x))] = x[i*step:min((i+1)*step, len(x))]

        result = fft_object()
        coeffs.append(np.copy(result[:maxk]))
        
    return np.stack(coeffs)

def get_magnitude(fourier):
    magnitude = np.abs(fourier)
    m = len(magnitude)
    magnitude2 = np.zeros_like(magnitude)
    magnitude2[:-(-m//2)] = magnitude[::2]
    magnitude3 = np.zeros_like(magnitude)
    magnitude3[:-(-m//3)] = magnitude[::3]
    magnitude4 = np.zeros_like(magnitude)
    magnitude4[:-(-m//4)] = magnitude[::4]
    
    return (magnitude + magnitude2 + magnitude3 + magnitude4)*get_bpm_prior()


def offset_from_fourier(fourier, idx, fps=50, num_windows=21, bpm_scale=0.1):
    return -1*np.angle(fourier[idx])/(2*np.pi*idx*bpm_scale/60) - 0.5/fps*(1-1/num_windows)

def get_bpm_offset(fourier, fps=50, num_windows=21, bpm_scale=0.1):
    max_mag = np.argmax(get_magnitude(fourier))
    bpm = max_mag*bpm_scale
    offset = offset_from_fourier(fourier, max_mag, fps=fps, num_windows=num_windows, bpm_scale=0.1)
    return bpm, offset

class BPMCost(rpt.base.BaseCost):
    model = ""
    min_size = 2
    
    def fit(self, signal):
        self.signal = np.asarray(signal)
        return self

    def error(self, start, end):
        segment = self.signal[start:end]
        n = len(segment)
        if n == 0:
            return 0
        else:
            return -np.max(get_magnitude(segment.sum(axis=0)))

    def sum_of_costs(self, indexes):
        return sum(self.error(start, end) for start, end in zip(indexes[:-1], indexes[1:]))

def get_segment_fourier(segments, fouriers, fps=50, num_windows=21):
    start_time = []
    bpm = []
    offset = []
    fourier = []
    start = 0
    for end in segments:
        n = end-start
        start_time.append(start)
        segment = fouriers[start:end]
        fourier_sum = np.sum(segment, axis=0)
        
        bpm_, offset_ = get_bpm_offset(fourier_sum, fps=fps, num_windows=num_windows, bpm_scale=0.1)
        bpm.append(bpm_)
        offset.append(offset_)
        fourier.append(fourier_sum)
        start = end
    
    return start_time, bpm, offset, fourier



def analyze_activation(beat_activation, num_windows=21, num_ensemble=3, sr=44100, fps=50, penalty=200, fourier_step=500):

    window_per_sec = fps * num_windows
    stepsize = fourier_step/window_per_sec

    beat_activation[np.isnan(beat_activation)] = 0

    beat_activations_synced = [np.repeat(np.array(beat_activation[i]), num_windows, axis=1)[:, i:-num_windows+i] for i in range(num_windows)]

    beat_array = np.array(beat_activations_synced)

    combined_beat_activation = (beat_array.prod(axis=1)**(1/num_ensemble)).prod(axis=0)**(1/num_windows)
    

    norm_activation = combined_beat_activation - combined_beat_activation.mean()
    
    print('performing DFT...')

    coeffs = dft_coeffs(norm_activation, N=int(window_per_sec*600), maxk=10000, step=fourier_step)

    print('Detecting changepoint...')
    algo = rpt.Pelt(custom_cost=BPMCost()).fit(coeffs)

    result = algo.predict(pen=penalty)

    start_time, bpm, offset, fourier = get_segment_fourier(result, coeffs, fps, num_windows)


    beats, _ = find_peaks(combined_beat_activation, prominence=0.2)
    beats = beats/window_per_sec - 0.5/fps*(1-1/num_windows)

    num_segments = len(fourier)
    combined_segments = [[i] for i in range(num_segments)]
    combined_fouriers = [fourier[i] for i in range(num_segments)]
    scores = np.zeros([num_segments, num_segments])

    for i in range(len(fourier)-1):
        for ii in range(i+1, len(fourier)):
            fourier_sum = fourier[i] + fourier[ii]
            score = np.max(get_magnitude(fourier_sum)) - np.max(get_magnitude(fourier[i])) - np.max(get_magnitude(fourier[ii])) + penalty 
            scores[i][ii] = score

    max_score = np.max(scores)

    while max_score > 0:
        i, ii = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        assert i < ii
        combined_segments[i] += combined_segments[ii]
        del combined_segments[ii]
        combined_fouriers[i] += combined_fouriers[ii]
        del combined_fouriers[ii]
        scores = np.delete(scores, ii, 0)
        scores = np.delete(scores, ii, 1)
        for j in range(0, i):
            fourier_sum = combined_fouriers[i] + combined_fouriers[j]
            score = np.max(get_magnitude(fourier_sum)) - np.max(get_magnitude(combined_fouriers[i])) - np.max(get_magnitude(combined_fouriers[j])) + penalty 
            scores[j][i] = score
        for j in range(i+1, len(combined_fouriers)):
            fourier_sum = combined_fouriers[i] + combined_fouriers[j]
            score = np.max(get_magnitude(fourier_sum)) - np.max(get_magnitude(combined_fouriers[i])) - np.max(get_magnitude(combined_fouriers[j])) + penalty 
            scores[i][j] = score
        max_score = np.max(scores)
        
    segment_beats = [[] for i in range(num_segments)]
    current_seg = 0
    for beat in beats:
        while beat > result[current_seg]*stepsize:
            current_seg += 1
        segment_beats[current_seg].append(beat)
    
    final_segments = []
    for i, idx_list in enumerate(combined_segments):
        fourier_ = combined_fouriers[i]
        bpm_, offset_ = get_bpm_offset(fourier_, fps=fps, num_windows=num_windows, bpm_scale=0.1)
        time_ = [[start_time[idx]*stepsize, result[idx]*stepsize] for idx in sorted(idx_list)]
        beats_ = [beat for idx in sorted(idx_list) for beat in segment_beats[idx]]
        
        beat_offsets_ = []
        for beat in beats_:
            beat_offset = (beat_+60/bpm_/2)%(60/bpm_) - 60/bpm_/2
            beat_offsets_.append(beat_offset)
        
        final_segments.append({'time': time_, 'bpm': bpm_, 'offset': offset_, 'beat':beats_, 'beat_offset':beat_offsets_, 'idx':sorted(idx_list)})
    
    return final_segments    



def color_map(row):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    if row['BPM'] == 0:
        return ['background-color: #7f7f7f']*len(row)
    else:
        return [f'background-color: {colors[int(row["ID"])%9]}']*len(row)

def show_data(results):
    segment_data = []
    for i, segment in enumerate(results):
        for ii, idx in enumerate(segment['idx']):
            segment_data.append({'idx':idx, 'Start':segment['time'][ii][0], 'End':segment['time'][ii][1], 'BPM':segment['bpm'], 'Offset':segment['offset']*1000, 'ID':i})
    segment_data = sorted(segment_data, key=lambda d: d['idx'])
    df = pd.DataFrame.from_dict(segment_data).drop('idx', axis=1)
    df[np.isinf(df)] = 0
    styled_df = df.style.apply(color_map, axis=1).format({'Start': '{:.1f}s', 'End': '{:.1f}s', 'BPM':'{:.1f}', 'Offset':'{:.1f} ms'})
    display(styled_df)


def plot_beats(results):
    import matplotlib.pyplot as plt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    width = 10.0
    height = width * 0.4
    fig = plt.figure(figsize = (width, height))
    ax = plt.gca()
    beats = np.array([beat for segment in results for beat in segment['beat']])
    offsets = np.array([beat_offset for segment in results for beat_offset in segment['beat_offset']])
    ax.scatter(beats, offsets*1000, zorder=8, alpha=1, s=3, c='black')
    color_idx = 0
    for segment in results:
        if segment['bpm'] == 0:
            color = '#7f7f7f'
        else:
            color = colors[color_idx%9]
            color_idx += 1
        for time in segment['time']:
            ax.axvspan(time[0], time[1], facecolor=color, alpha=0.2)
            if segment['bpm'] != 0 and not np.isinf(segment['offset']) and time[1]-time[0]>10:
                ax.text(time[0], 1.1, f'{segment["bpm"]:.1f}', horizontalalignment='left', verticalalignment='bottom', 
                    transform=ax.get_xaxis_transform(), fontsize=8)
                ax.text(time[0], 1.03, f'{segment["offset"]*1000:.1f}', horizontalalignment='left', verticalalignment='bottom', 
                    transform=ax.get_xaxis_transform(), fontsize=8)
    ax.text(-0.06, 1.1, f'BPM', horizontalalignment='left', verticalalignment='bottom', fontsize=8, transform = ax.transAxes)
    ax.text(-0.06, 1.03, f'Offset', horizontalalignment='left', verticalalignment='bottom', fontsize=8, transform= ax.transAxes)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('offset (ms)')
    ax.set_ylim(-60, 60)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    
    ax.grid(which='both')
    ax.set_xlim(0)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.show()
    
