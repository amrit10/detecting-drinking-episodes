import os
import pandas as pd
import numpy as np
import datetime
import scipy
from scipy.stats import skew, kurtosis
     
os.listdir('../data/clean_tac')

base_path = '../data/clean_tac'
my_dict = dict()
for file_path in os.listdir(base_path):
  resp_frame = pd.read_csv(base_path + '/'+ file_path)
  my_dict[file_path.split('_')[0]] = resp_frame

def get_tac_value(pid, t_value):
  ind = np.argmax(my_dict[pid]['timestamp'] > t_value)
  if ind != 0:
    ind = ind - 1
  return my_dict[pid].iloc[[ind]]['TAC_Reading'].values[0]

def spectral_centroid_spread(fft_magnitude, sampling_rate):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(fft_magnitude) + 1)) * \
          (sampling_rate / (2.0 * len(fft_magnitude)))
    eps = 0.00000001
    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps
 
    # Centroid:
    centroid = (NUM / DEN)
 
    # Spread:
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)
 
    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)
    spread = spread / (sampling_rate / 2.0)
 
    return centroid
def energy_entropy(frame, n_short_blocks):
    """Computes entropy of energy"""
    # total frame energy
    eps = 0.00000001
    for i in frame:
      frame_energy = np.sum(i ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]
 
    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()
 
    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)
 
    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy

def energy(frame):
  return np.sum(frame ** 2) / np.float64(len(frame))


def transform_frames(f, pid):
  cols = ['x', 'y', 'z']
  funcs = [('mean', np.mean), ('variance', np.var), ('median', np.median), ('min', np.min), ('max', np.max), ('rms', lambda x: np.sqrt(np.mean(np.square(x)))), ('energy_entropy', lambda x: energy_entropy(x, 10)), ('energy', energy), ('skew', skew), ('Kurtiosis', kurtosis)]
  fft_funcs = [('variance', np.var), ('spectral_centroid_spread', lambda x: spectral_centroid_spread(x, 40)) ]
  new_frame = pd.DataFrame()
  f['time_second'] = f['time'].apply(get_val)
  some_values = f.groupby('time_second')
  new_frame['TAC_reading'] = f.groupby('time_second')['x'].apply(np.array).index.to_series().apply(lambda x: get_tac_value(pid, x))
  for col in cols:
    col_array = some_values[col].apply(np.array)
    col_fft = col_array.apply(scipy.fft.fft)
    for key, func in funcs:
      print(pid,' :Working on', col, '   ', key)
      new_frame['_'.join([col, key])] = col_array.apply(func)
    for key, func in fft_funcs:
      print(pid,' :Working on', col, '   FFT ', key)
      new_frame['_'.join([col, 'FFT', key])] = col_fft.apply(func)
  new_frame['pid'] = pid
  print(new_frame.head())
  return new_frame

def get_time_value(x):
    t = datetime.datetime.fromtimestamp(x/1000.0)
    t = t.replace(microsecond = 0)
    return t.timestamp()
def get_time_ignore_second(x):
    t = datetime.datetime.fromtimestamp(x/1000.0)
    t = t.replace(microsecond = 0)
    t = t.replace(second = int(t.second / 10))
    return t.timestamp()
def get_window_10(x):
    t = datetime.datetime.fromtimestamp(x)
    t = t.replace(second = int(t.second / 10))
    return t.timestamp()
     

frame = pd.read_csv('/content/drive/My Drive/CSE512Data/all_accelerometer_data_pids_13.csv')
frame['window'] = frame['time'].apply(get_time_value)
frame['window10'] = frame['time'].apply(get_time_ignore_second)
frame['selected_window_10'] = frame['window'].apply(get_window_10)
     

funcs = [('mean', np.mean), ('variance', np.var), ('median', np.median), ('min', np.min), ('max', np.max), ('rms', lambda x: np.sqrt(np.mean(np.square(x)))), ('energy_entropy', lambda x: energy_entropy(x, 10)), ('energy', energy), ('skew', skew), ('Kurtiosis', kurtosis)]
fft_funcs = [('variance', np.var), ('spectral_centroid_spread', lambda x: spectral_centroid_spread(x, 40)) ]
cols = ['x', 'y', 'z']
     

look_up_frames = dict()
for pid in frame.pid.unique():
  response_frame = pd.DataFrame()
  grouped = frame[frame.pid == pid].groupby('window10')
  for col in cols:
    col_array = grouped[col].apply(np.array)
    col_fft = col_array.apply(scipy.fft.fft)
    for key, func in funcs:
      print(pid,' :Working on', col, '   ', key)
      response_frame['_'.join(['win_10', col, key])] = col_array.apply(func)
    for key, func in fft_funcs:
      print(pid,' :Working on', col, '   FFT ', key)
      response_frame['_'.join(['win_10', col, 'FFT', key])] = col_fft.apply(func)
  response_frame.pid = pid
  look_up_frames[pid] = response_frame
