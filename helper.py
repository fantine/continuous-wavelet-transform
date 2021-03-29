import numpy as np

__all__ = ["next_p2", "pad"]

def next_p2(n):
    """Returns the smallest power of 2 greater than n"""

    if n < 1:
        raise ValueError("n must be >= 1")

    return 1 << (n-1).bit_length()



def pad(x, method='reflection'):
    """Pad to bring the total length N up to the next-higher
    power of two.

    Args: 
       x (1d ndarray): data
       method (str): 'reflection', 'periodic' or 'zeros'
       
    Returns:
       xp, orig (1d ndarray, 1d ndarray boolean):
          padded version of x and a boolean array with
          value True where xp contains the original data
    """

    x_arr = np.asarray(x)

    if not method in ['reflection', 'periodic', 'zeros']:
        raise ValueError('Unavailable padding method')

    diff = next_p2(x_arr.shape[0]) - x_arr.shape[0]
    ldiff = int(diff / 2)
    rdiff = diff - ldiff

    if method == 'reflection':
        left_x = x_arr[:ldiff][::-1]
        right_x = x_arr[-rdiff:][::-1]
    elif method == 'periodic':
        left_x = x_arr[:ldiff]
        right_x = x_arr[-rdiff:]
    elif method == 'zeros':
        left_x = np.zeros(ldiff, dtype=x_arr.dtype)
        right_x = np.zeros(rdiff, dtype=x_arr.dtype)

    xp = np.concatenate((left_x, x_arr, right_x))
    orig = np.ones(x_arr.shape[0] + diff, dtype=np.bool)
    orig[:ldiff] = False
    orig[-rdiff:] = False

    return xp, orig



from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# def dict_to_csv(dic, args, filename):
#     """
#     Saves the dict as a .csv file.
#     Args:
#         dic     (str):  The dict to be saved as .csv
#         args    (arg):  The args passed to main to be able to access save location
#         filename(str):  Name of the .csv file
#     Returns:
#     """
#     sorted_labels = sorted(list(dic.items()), key=lambda x: int(x[0].split('-')[-1]))
#     if args.v:
#         print('[INFO] Annotations for following images exist:')
#         print(['{0:g}'.format(int(x[0].split('-')[-1])) for x in sorted_labels])
#     with open(os.path.join(LABELSPATH, filename), 'w') as f:
#         [f.write('{0},{1}\n'.format(k, v)) for k, v in sorted_labels]
 
import csv

def csv_to_arr(filepath, n):
    """
    Reads the .csv file into an array
    Args:
        filepath    (str) : Path to the .csv file to be read in
    Returns:
        arr         (1d ndarray): labels
    """
    
    labels = np.zeros(n)
    
    with open(filepath, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader: 
            # dic = {rows[0]:int(rows[1]) for rows in reader}
            index = int(rows[0][2:])
            val = int(rows[1][3:])
            labels[index -1] = val
    
    return labels