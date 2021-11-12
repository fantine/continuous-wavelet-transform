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
