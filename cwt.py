import numpy as np

__all__ = ["cwt", "icwt", "cwt2d", "icwt2d", "dyadic_scales", "freq_from_scales",
           "scales_from_freq"]

PI2 = 2 * np.pi


### BEGIN STANFORD CODE ###

def angular_freq(N):
  """Compute angular frequencies

  Args:
    N (int): number of samples

  Returns:
    angular frequencies (1d ndarray)

  These angular frequencies are independent of 
  the physical sampling rate of the original data. 
  They represent an abstract discretization of
  the wavelet space. 
  """

  w = np.empty(N)

  nyq = N // 2

  for i in range(w.shape[0]):
    if i <= nyq:
      # Positive frequencies
      w[i] = PI2 * i / N
    else:
      # Negative frequencies
      w[i] = PI2 * (i - N) / N

  return w



def dyadic_scales(N, dj=0.25, wavelet='morlet', w0=6):
  """Compute scales as fractional powers of two

  Args:
    N (int): number of samples
    dj (float): scale resolution
      1 / dj corresponds to the number of voices per octave
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter

  Returns:
    scales (1d ndarray)

  These dyadic scales are independent of 
  the physical sampling rate of the original data. 
  They correspond to an abstract discretization of
  the wavelet space.
  Their corrresponding physical frequencies or wavenumbers
  can be computed using the function freq_from_scales.
  """

  if wavelet == 'morlet':
    # Smallest resolvable scale
    s0 = 2 * (w0 + np.sqrt(2 + w0**2)) / (4 * np.pi)
  else:
    raise ValueError('wavelet function not available')

  J = int(np.floor(np.log2(N / s0) / dj))
  s = np.empty(J + 1)

  for i in range(s.shape[0]):
    s[i] = s0 * 2**(i * dj)

  return s



def freq_from_scales(s, dt=1, wavelet='morlet', w0=6):
  """Compute frequencies or wavenumbers from abstract scales

  Args:
    s (1d ndarray): scales
    dt (float): sampling rate
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter

  Returns:
    Frequencies or wavenumbers (1d ndarray)

  As the scales correspond to an abstract discretization of
  the wavelet space, this function allows for re-mapping them 
  into the physical space of frequencies and wavenumbers.
  """

  s_arr = np.asarray(s)

  if s_arr.ndim is not 1:
    raise ValueError('s (scales) must be an 1d numpy array')

  if wavelet == 'morlet':
    return  (w0 + np.sqrt(2 + w0**2)) / (4 * np.pi * dt * s_arr)
  else:
    raise ValueError('wavelet function not available')

    

def scales_from_freq(f, dt=1, wavelet='morlet', w0=6):
  """Compute abstract scales from frequencies or wavenumbers

  Args:
    f (1d ndarray): frequencies or wavenumbers
    dt (float): sampling rate
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter

  Returns:
    scales (1d ndarray)

  As the scales correspond to an abstract discretization of
  the wavelet space, this function allows for computing them 
  from the physical frequencies and wavenumbers.
  """

  f_arr = np.asarray(f)

  if f_arr.ndim is not 1:
    raise ValueError('f (frequencies or wavenumbers) must be an 1d numpy array')

  if wavelet == 'morlet':
    return (w0 + np.sqrt(2 + w0**2)) / (4 * np.pi * dt * f_arr)
  else:
    raise ValueError('wavelet function not available')



def morlet1d_ft(s, w, w0=6):
  """Fourier tranformed scaled 1D Morlet wavelet dictionary

  Args:
    s (1d ndarray): scales
    w (1d ndarray): angular frequencies
    w0 (float): Morlet wavelet frequency parameter
      Trade-off between temporal and frequency precision

  Returns:
    Fourier transformed scaled Morlet wavelet dictionary
      of shape (n_scales, n_angular_frequencies)

  It is recommended to pick a value of w0 greater than 6 so
  that the wavelet can be approximated as an analytic wavelet
  """

  psi = np.zeros((s.shape[0], w.shape[0]))

  # Approximate as an analytic wavelet
  pos = w > 0

  c = (4 * np.pi)**-0.25

  for i in range(s.shape[0]):
      psi[i][pos] = np.sqrt(s[i]) * c * np.exp(- (s[i] * w[pos] - w0)**2 / 2)

  return psi



def cwt1d(x, s, dt=1, wavelet='morlet', w0=6):
  """Continuous Wavelet Tranform in 1D

  Args:
    x (1d ndarray): input data (of length n_axis1)
    s (1d ndarray): scaling factors
    dt (float): sampling rate
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter

  Returns:
    X (2d ndarray): tranformed data
      of shape (n_scales, n_axis1)
  """

  x_arr = np.asarray(x)
  s_arr = np.asarray(s)

  if x_arr.ndim is not 1:
    raise ValueError('x (input data) must be an 1d numpy array')

  if s_arr.ndim is not 1:
    raise ValueError('s (scales) must be an 1d numpy array')

  w = angular_freq(x.shape[0])

  if wavelet == 'morlet':
    psi = morlet1d_ft(s_arr, w, w0)
  else:
    raise ValueError('wavelet function is not available')
  
  # The array that will contain the transformed data of shape (n_scales, n_axis1)
  X_ARR = np.empty((psi.shape[0], psi.shape[1]), dtype='complex128')

  # TODO replace numpy fft with a better FFT implementation
  # TODO add appropriate padding and tapering to ensure that the signal is periodical
  x_arr_ft = np.fft.fft(x_arr)

  # Perform the circular convolution in Fourier domain
  for i in range(X_ARR.shape[0]):
    # Even if the Fourier transform of the Morlet wavelet is real, 
    # keep the conjugate here, so as to allow other wavelets
    X_ARR[i] = np.fft.ifft(x_arr_ft * np.conj(psi[i]))

  # TODO Write helper function to calculate adjustment coefficient for all n_axis1 and dt

  return X_ARR



def icwt1d(X, s, dt=1, wavelet='morlet', w0=6):
  """Inverse Continuous Wavelet Tranform in 1D

  Args:
    X (2d ndarray): tranformed data
      of shape (n_scales, n_axis1)
    s (1d ndarray): scaling factors
    dt (float): sampling rate
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter

  Returns:
    x (1d ndarray): reconstructed data
      of length n_axis1
  """

  X_arr = np.asarray(X)
  s_arr = np.asarray(s)

  if X_arr.shape[0] != s_arr.shape[0]:
    raise ValueError('X, scales: shape mismatch')

  if X_arr.ndim is not 2:
    raise ValueError('X (transformed data) must be an 2d numpy array')

    
  if s_arr.ndim is not 1:
    raise ValueError('s (scales) must be an 1d numpy array')

  # Default Python behavior is to pass on pointers for arrays
  # Therefore, we have to create a new array in order not to mess up 
  # the input data 
  X_ARR = np.empty_like(X_arr)
    
  # The reconstruction can be performed as a weighted summation
  for i in range(s_arr.shape[0]):
    X_ARR[i] = X_arr[i] / s_arr[i]**0.5
    
  # TODO Write helper function to calculate reconstruction coefficient for all n_axis1, ds, and dt
  c = 1.51749474811
  x = c * np.sum(np.real(X_ARR), axis=0)

  return x

### END STANFORD CODE ###



### BEGIN SLB CODE ###

def morlet2d_ft(s, thetas, w1, w2, w0=6):
  """Fourier tranformed scaled 2D Morlet wavelet dictionary

  Args:
    s (1d ndarray): scales
    thetas (1d ndarray): rotation angles of the wavelet
    w1 (1d ndarray): angular frequencies along the first axis
    w2 (1d ndarray): angular frequencies along the second axis
    w0 (float): Morlet wavelet frequency parameter
      Trade-off between temporal and frequency precision

  Returns:
    Fourier transformed scaled Morlet wavelet dictionary
      of shape (n_scales, n_thetas, n_angular_freq_1, n_angular_freq_2)
  """

  psi = np.empty((s.shape[0], thetas.shape[0], w1.shape[0], w2.shape[0]))
  kx, ky = np.meshgrid(w2, w1)

  for j, theta in enumerate(thetas): 
      theta = - theta # inverse of the rotation matrix
      
      # Apply the rotation matrix
      kxr = (kx * np.cos(theta) + ky * np.sin(theta))
      kyr = (ky * np.cos(theta) - kx * np.sin(theta))

      for i in range(s.shape[0]):
        kxri = s[i] * kxr
        kyri = s[i] * kyr

        kr2 = kxri**2 + kyri**2
        krw02 = (kxri - w0)**2 + kyri**2
        
        psi[i, j] = PI2 * s[i] * (np.exp(- 0.5 * krw02) - np.exp(- 0.5 * w0**2) * np.exp(- 0.5 * kr2)) 
        
  return psi



def cwt2d(x, s, thetas, d1, d2, wavelet='morlet', w0=6):
  """Continuous Wavelet Tranform in 2D

  Args:
    x (2d ndarray): input data of shape (n_axis1, n_axis2)
    s (1d ndarray): scaling factors
    thetas (1d ndarray): rotation angles of the wavelet
    d1 (float): sampling rate on axis 1 
    d2 (float): sampling rate on axis 2
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter   

  Returns:
    X (4d ndarray): transformed data 
      of shape (n_scales, n_thetas, n_axis1, n_axis2)
  """

  x_arr = np.asarray(x)
  s_arr = np.asarray(s)
  th_arr = np.asarray(thetas)

  if x_arr.ndim is not 2:
    raise ValueError('x (input data) must be an 2d numpy array')

  if s_arr.ndim is not 1:
    raise ValueError('s (scales) must be an 1d numpy array')
    
  if th_arr.ndim is not 1:
    raise ValueError('thetas (rotation angles) must be an 1d numpy array')
    
  n1, n2 = x_arr.shape
  w1 = angular_freq(n1)
  w2 = angular_freq(n2)

  if wavelet == 'morlet':
    psi = morlet2d_ft(s_arr, thetas, w1, w2, w0)
  else:
    raise ValueError('wavelet function is not available')

  # The array that will contain the transformed data of shape (n_scales, n_thetas, n_axis1, n_axis2)
  X_ARR = np.empty((s_arr.shape[0], th_arr.shape[0], n1, n2), dtype='complex128')
    
  # TODO replace numpy fft with a better FFT implementation
  # TODO add appropriate padding and tapering to ensure that the signal is periodical
  x_arr_ft = np.fft.fft2(x_arr)

  # Perform the circular convolution in Fourier domain
  for i in range(s_arr.shape[0]):
    for j in range(th_arr.shape[0]):
      # Even if the Fourier transform of the Morlet wavelet is real, 
      # keep the conjugate here, so as to allow other wavelets
      X_ARR[i,j,:,:] = np.fft.ifft2(x_arr_ft * np.conj(psi[i,j]))

  # TODO Write helper function to calculate adjustment coefficient for all n_axis1, n_axis2, d1 and d2
        
  return X_ARR



def icwt2d(X, s, thetas, d1, d2, wavelet='morlet', w0=6):
  """Inverse Continuous Wavelet Transform in 2D

  Args:
    X (4d ndarray): tranformed data
      of shape (n_scales, n_thetas, n_axis1, n_axis2)
    s (1d ndarray): scaling factors 
    thetas (1d ndarray): rotation angles of the wavelet
    d1 (float): sampling rate on axis 1 
    d2 (float): sampling rate on axis 2
    wavelet (str): name of the wavelet function ('morlet')
    w0 (float): wavelet parameter   

  Returns:
    x (2d ndarray): reconstructed data
      of shape (n_axis1, n_axis2)
  """

  X_arr = np.asarray(X)
  s_arr = np.asarray(s)
  th_arr = np.asarray(thetas)

  if X_arr.shape[0] != s_arr.shape[0]:
    raise ValueError('X, scales: shape mismatch')

  if X_arr.shape[1] != th_arr.shape[0]:
    raise ValueError('X, thetas: shape mismatch')

  if X_arr.ndim is not 4:
    raise ValueError('X (transformed data) must be an 4d numpy array')

  if s_arr.ndim is not 1:
    raise ValueError('s (scales) must be an 1d numpy array')
    
  if th_arr.ndim is not 1:
    raise ValueError('thetas (rotation angles) must be an 1d numpy array')


  # Default Python behavior is to pass on pointers for arrays
  # Therefore, we have to create a new array in order not to mess up 
  # the input data 
  X_ARR = np.empty_like(X_arr)
    
  # The reconstruction can be performed as a weighted summation
  for i in range(s_arr.shape[0]):
    X_ARR[i] = X_arr[i] / s_arr[i]

  # TODO Write helper function to calculate reconstruction coefficient for all n_axis1, n_axis2, ds, d1 and d2
  c = 1.98036746

  x = c * np.sum(np.sum(np.real(X_ARR), axis=1), axis=0)

  return x

### END SLB CODE ###