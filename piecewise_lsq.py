# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:27:29 2024

Piecewise polynomial continuous least-square fit of 1D data

An input data (including noise or undesired high-frequency perturbations) can
be filtered by computing polynomial least-square fits over multiple windows,
smoothing the input data.
Continuity of some of the derivatives of the polynomials across the window boundaries
can be enforced.

This seems to be more accurate than a Savitzky-Golay filter in some cases,
even though the latter is also based on a local least-square fit, without
continuity constraints however.

Some information may be found on this topic :
  http://www.seas.ucla.edu/~vandenbe/133A/lectures/cls.pdf
  https://galton.uchicago.edu/~lekheng/courses/302/notes12.pdf
  
@author: lfrancoi
"""
import numpy as np
factorial = np.math.factorial

class Interpolant():
  """ Polynomial interpolant for a single interval """
  def __init__(self, ta, tb, coeffs, tmid):
    self.ta = ta
    self.tb = tb
    self.coeffs = coeffs
    self.tmid = tmid
    
  def __call__(self, t, der=0):
    """ Evaluate the polynomial at time t (float or np.ndarray) """
    assert np.all(t>=self.ta)
    assert np.all(t<=self.tb)
    poly = t*0.
    for k in range(der,len(self.coeffs)):
      poly += self.coeffs[k] * factorial(k)/factorial(k-der) * (t - self.tmid )**(k-der)
    return poly
      
class PLSQC():
  """ Piecewise Least-SQuare polynomial fit with Continuity constraints
  
  Upon construction, the constrained least-square problem is solved, and internal
  data structures are built and filled in, so that the evaluation of the filtered
  signal and its derivatives are possible.
  
    Parameters
    ----------
    x : (N,) array-like
      Sampling times for the filtered signal. Must be sorted in ascending order.
      The sampling can be nonuniform, i.e. with a non-constant sampling step.
    y : (N,) array-like
      Sampled signal
    deg : int, optional
      Degree of the polymial constructed in each window. The default is 3.
    f : float, optional
      Low-pass cutting frequency. The values of f and T and cannot be both
      specified, since f=1/T. The default is None, i.e. computed from T.
    T : float, optional
      Filtering window size. The default is None, i.e. computed from f.
    continuity : int, array_like, optional
      The derivatives of the fitted polynimials must be continuous at the boundary
      between consecutive windows. If continuity is an int, all derivatives up
      to the one of order continuity will be continuous. If it is an array, tuple,
      or list, the values should represent the derivatives which must be continuous.
      This can allow for a continuous second-order derivative to be enforce,
      while keeping the first-order derivative potentially discontinuous.
      The default is deg-1.
    overlap : float, optional
      The proportion by which each window is extended on each side to overlap with
      the neighbouring windows. The default is 0.5.
    scale_with_overlap : logical, optional
      If true, the window length is adjusted internally so that it is kept at its
      original value once the overlap extension has been performed. This allows the
      effective filtering frequency to be maintained for any amount of overlap.
      The default is True.
    debug : logical, optional
      If true, printouts and plots are made to test various aspects of the code.
      The default is False.

    Returns
    -------
    None.

  """
  
  def __init__(self, x, y, deg=3, f=None, T=None, continuity=None,
               overlap=0.5, scale_with_overlap=True, debug=False):
    self.x = x # data sampling point
    assert np.all(np.diff(x))>0, 'x must be sorted'
    self.y = y # noisy data
    
    assert deg >= 0, 'deg must be >= 0'
    assert isinstance(deg, int), 'deg must be an int'
    self.deg = deg # degrees of the polynomials
    
    if continuity is None:
      self.continuity = np.array(range(deg)) # up to deg-1
    elif isinstance(continuity, int):
      self.continuity = np.array(range(continuity+1))
    else:      
      self.continuity = np.unique(continuity) # list of the derivatives which must be continuous
      assert self.continuity.ndim==1, 'Continuity must be a 1D array, a tuple, or an int'
      
    assert max(self.continuity)<self.deg, \
        "Required continuity for the derivatives is larger than or equal to the degree"
        
    if not (f is None):
      assert T is None, 'f and T cannot be both prescribed'
      self.f = f # cutting frequency
      self.T = 1./f # window length
    elif not (T is None):
      self.T = T
      self.f = 1./T
    else:
      raise ValueError("Either the cut frequency f of the window length T (time) must be prescribed")
    
    # assert overlap <= 1, 'overlap must be <= 1' # not actually necessary
    # assert overlap >= 0, 'overlap must be >= 0' # not actually necessary, but makes sense
    self.overlap = overlap # portion by which consecutive windows overlap
    self.scale_with_overlap = scale_with_overlap
    if self.scale_with_overlap:
      self.T = self.T /(2 + self.overlap)
      self.f = 1/self.T
    
    self.debug = debug
    self._solve_constrained_lsq() # compute coefficients
    self._construct_interpolants() # construct the interpolants for each interval
  
  def _construct_interpolants(self):
    """ Construct the polynomial interpolant for each interval """
    self.interpolants = []
    istart = 0
    for i in range(len(self.times)-1): # go through each interval
      self.interpolants.append( Interpolant(ta=self.times[i], tb=self.times[i+1],
                                            tmid=(self.times[i]+self.times[i+1])/2,
                                            coeffs=self.coeffs[istart:istart+self.deg+1]) )
      istart += self.deg + 1
      
  def __call__(self, xi, der=0):
    """ Evaluate PLSQC polynomials on the xi grid
       Adapted from scipy.integrate.solve_ivp's dense output evaluation. """
    t = np.asarray(xi)
    self.n_segments = len(self.times) - 1
    
    if t.ndim == 0:
        ind = np.searchsorted(self.times, t, side='left')
        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment
        return self.interpolants[segment](t, der=der)
    else:
      order = np.argsort(t)
      reverse = np.empty_like(order)
      reverse[order] = np.arange(order.shape[0])
      t_sorted = t[order]
  
      segments = np.searchsorted(self.times, t_sorted, side='left')
      segments -= 1
      segments[segments < 0] = 0
      segments[segments > self.n_segments - 1] = self.n_segments - 1
      ys = []
      group_start = 0
      from itertools import groupby
      for segment, group in groupby(segments):
          group_end = group_start + len(list(group))
          y = self.interpolants[segment](t_sorted[group_start:group_end], der=der)
          ys.append(y)
          group_start = group_end
  
      ys = np.hstack(ys)
      ys = ys[reverse]
    return ys
  
  def _solve_constrained_lsq(self):
    """ Constructs the matrix representing the LSQ problems with continuity constraints,
    and solves it to determine the coefficients of the polynomials on each interval"""
    
    # 1 - Determine windows
    times = np.arange(self.x[0], self.x[-1], self.T)
    if abs(times[-1] - self.x[-1])/self.T < 1e-3: # not close enough to the end
      times[-1] = self.x[-1]
    else:
      times = np.hstack((times, self.x[-1]))
    
    n = times.size - 1 # number of windows
    if self.debug: print(f'{n} windows')
    
    # 2 - Construct the vector Y of function values to match in a lesat-squre sense,
    # potentially including repeated values for the overlaps.
    Y = []
    # np.searchsorted(self.x, times, side='left', sorter=None)
    X = [] # times at which the values Y corresponds
    Mblocks = [] # each block of the block-diagonal M matrix
    xcenters = []
    times_extended = []
    for i in range(n): # loop on each window
      ti = times[i] # left boundary
      tip1 = times[i+1] # right boundary
      if i==n-1: # specific handling for the last window
        # ensure that it has the same length as the others
        tip1 = times[i+1]
        dt = times[1]-times[0]
        ti = tip1 - dt
        
      tip1o2 = 0.5*(ti + tip1) # center of the window
      dt = tip1 - ti # length of the window
      
      # correct for overlap
      ti   = tip1o2 - dt/2 - dt*self.overlap
      tip1 = tip1o2 + dt/2 + dt*self.overlap
      # Interval i overlaps on by a proportion self.overlap on the next interval.
      # Overall, they overlap by twice this proportion once they have been both extendeded.
      imin = np.searchsorted(self.x, ti,   side='left', sorter=None)
      imax = np.searchsorted(self.x, tip1, side='right', sorter=None) - 1
      assert self.x[imin]>=ti
      assert self.x[imax]<=tip1
      Y.append(self.y[imin:imax+1])
      X.append(self.x[imin:imax+1])
      assert imax - imin > self.deg + 1 , 'not enough data points' # fit would be exact
      xcenters.append(tip1o2)
      times_extended.append(ti)
      
      # construct matrix block
      Mblock = np.zeros((Y[-1].size, self.deg+1))
      for i in range(Y[-1].size):
        for k in range(self.deg+1):
          Mblock[i,k] = ( X[-1][i] - tip1o2 )**k
        
      Mblocks.append(Mblock)
      
    Yr = np.hstack(Y)
    # Xr = np.hstack(X)
    
    import scipy.sparse
    M = scipy.sparse.block_diag(Mblocks)
    
    if self.debug:
        # test without constraint: M coeffs = Y in least-square sense
        import scipy.linalg
        Mpinv = scipy.linalg.pinv(M.toarray())
        coeffs = Mpinv @ Yr
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.x, self.y, label='data', color='tab:blue')
        istart = 0
        for i in range(n):
          poly = X[i]*0.
          for k in range(self.deg+1):
            poly += coeffs[istart+k] * (X[i]-xcenters[i])**k
          istart += self.deg + 1
          plt.plot(X[i], poly, color='tab:red', label=None)
          # compare with Numpy's lsq fit
          polynp = np.polyfit( X[i], Y[i], deg=self.deg)
          plt.plot(X[i], np.polyval(polynp, X[i]), color='tab:red',
                   label=None, linestyle='--', linewidth=3)
    
        for i in range(n+1):
          plt.axvline(times[i], linestyle=':', color=[0,0,0, 0.5])
        plt.plot(np.nan, color='tab:red', label='fit')
        plt.plot(np.nan, color='tab:red',linestyle='--', label='numpy polyfit')
        plt.legend()
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Test without constraints')
        
    #%% Construct constraints matrix
    
    # for each connection point between consecutive intervals (excluding the overlap),
    # we enforce the continuity of the self.continuity-th derivatives
    # It is possible to enforce only the 0-th and 2-nd derivative to be continuous,
    # letting the 1st be discontinuous. Who knows who might need that ? :)
    nc = len(self.continuity)
    if nc>0:
      C = np.zeros(( nc * (n-1), n*(self.deg+1) ))
      for i in range(n-1):
        tmid = times[i+1]
        tip1o2 = (times[i+1] +   times[i])*0.5
        tip3o2 = (times[i+2] + times[i+1])*0.5
        diagblock     = np.zeros(( nc, self.deg+1))
        for im, m in enumerate(self.continuity):
          for k in range(m,self.deg+1):
            diagblock[im,k] = factorial(k)/factorial(k-m) * (tmid - tip1o2 )**(k-m)

        neighborblock = np.zeros(( nc, self.deg+1))
        for im, m in enumerate(self.continuity):
          for k in range(m,self.deg+1):
            neighborblock[im,k] = -factorial(k)/factorial(k-m) * (tmid - tip3o2 )**(k-m)
        C[i*nc:(i+1)*nc, i*(self.deg+1):(i+1)*(self.deg+1)] = diagblock
        C[i*nc:(i+1)*nc, (i+1)*(self.deg+1):(i+2)*(self.deg+1)] = neighborblock
        
    # Assemble the overall matrix
    global_matrix = np.zeros((n*(self.deg+1) + nc*(n-1), n*(self.deg+1)+nc*(n-1)))
    # import pdb; pdb.set_trace()
    tmp = M.T @ M
    global_matrix[:n*(self.deg+1),:n*(self.deg+1)] = tmp.toarray()
    global_matrix[n*(self.deg+1):,:n*(self.deg+1)] = C
    global_matrix[:n*(self.deg+1),n*(self.deg+1):] = C.T
    
    if self.debug:
      import matplotlib.pyplot as plt
      plt.figure()
      plt.spy(global_matrix)
      plt.grid()
      plt.title('Global matrix')
      
    b = np.zeros((n*(self.deg+1) + nc*(n-1),))
    b[: n*(self.deg+1)] = M.T @ Yr
    sol = np.linalg.solve(global_matrix, b)
    
    coeffs = sol[: n*(self.deg+1)]
    # z = sol[n*(self.deg+1):] # lagrange multipliers
    
    if self.debug:
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot(self.x, self.y, label='data', color='tab:blue')
      istart = 0
      for i in range(n):
        poly = X[i]*0.
        for k in range(self.deg+1):
          poly += coeffs[istart+k] * (X[i]-xcenters[i])**k
        istart += self.deg + 1
        plt.plot(X[i], poly, color='tab:red', label=None)
        # compare with Numpy's lsq fit
        # polynp = np.polyfit( X[i], Y[i], deg=self.deg)
        # plt.plot(X[i], np.polyval(polynp, X[i]), color='tab:red',
        #          label=None, linestyle='--', linewidth=3)

      # compare with Numpy's lsq fit
      # polynp = np.polyfit( self.x, self.y, deg=self.deg)
      # plt.plot(self.x, np.polyval(polynp, self.x), color='tab:red',
      #           label='np.polyfit global', linestyle='--', linewidth=3)
      for i in range(n+1):
        plt.axvline(times[i], linestyle=':', color=[0,0,0, 0.5])
      plt.plot(np.nan, color='tab:red', label='fit')
      # plt.plot(np.nan, color='tab:red',linestyle='--', label='numpy polyfit')
      plt.legend()
      plt.grid()
      plt.xlabel('x')
      plt.ylabel('y')
      plt.title('Result with constraints')
      plt.ylim(-1.5, 1.5)
      plt.xlim(0,1)
      
    # store results
    self.coeffs = coeffs
    self.coeff_list = coeffs.reshape((n,self.deg+1), order='C')
    assert np.all(self.coeffs[:self.deg+1] == self.coeff_list[0,:])
    self.times = times
      
      
if __name__=='__main__':
  #%% Test and compare with other filters
  
  from scipy.signal import savgol_filter
  import matplotlib.pyplot as plt
  import numpy.random

  f_signal = 1. # frequency of the main signal
  f_noise = 30. # frequency of the noise to be filtered
  a_signal = 1. # signal amplitude
  a_noise = 0.3 # noise amplitude
  a_rand = 0.3 # amplitude of the random noise
  
  x = np.linspace(0, 1/f_signal, 20000) # sampling times
  y =   a_signal*np.cos(2*np.pi*f_signal*x) \
      + a_noise*np.cos(2*np.pi*f_noise*x)   \
      + a_rand*2*(np.random.rand(x.size)-0.5) # sampled signal
      
  dx = x[1]-x[0] # grid spacing for Savgol filter
  f_cut = 5. # low-pass cutting frequency
  T = 1/f_cut # window size
  # T = 0.1 # window size
  deg = 3 # which polynomial degree to use
  
  # instantiante PLSQC filter
  obj = PLSQC(x=x, y=y, T=T, continuity=2, deg=deg,
             overlap=1., debug=False, scale_with_overlap=True)
    
    
  # Compared the filtered result and the input
  plt.figure()
  plt.plot(x, y, label='input')
  plt.plot(x, obj(x), label='PLSQC', linestyle='--')
  plt.plot(x, savgol_filter(y, window_length=2*(int(T/dx)//2)+1, polyorder=deg,
                            deriv=0, delta=dx,
                            mode='interp', cval=0.0),
            label='Savitzky-Golay filter', linestyle='--', linewidth=2)
  plt.legend()
  plt.grid()
  plt.xlabel(r'$x$')
  plt.ylabel(r'$y$')
  plt.ylim(-1.5, 1.5)
  plt.xlim(x[0], x[-1])
  plt.title('Original and filtered signal')
  plt.savefig('img/filtering.png', dpi=200)

  # Compare the obtained filtered derivatives to the exact ones, and to those obtained with
  # other filters
  true_der = [lambda x:  np.cos(2*np.pi*f_signal*x),
              lambda x: -2*np.pi*f_signal * np.sin(2*np.pi*f_signal*x),
              lambda x: -(2*np.pi*f_signal)**2 * np.cos(2*np.pi*f_signal*x),
              lambda x:  (2*np.pi*f_signal)**3 * np.sin(2*np.pi*f_signal*x)]

  for der in range(4):
    plt.figure()
    plt.plot(x, true_der[der](x), label='input')
    ylims = plt.ylim()
    plt.plot(x, savgol_filter(y, window_length=2*(int(T/dx)//2)+1, polyorder=deg,
                              deriv=der, delta=dx,
                              mode='interp', cval=0.0),
              label='Savitzky-Golay filter', linestyle='--', linewidth=2)
    plt.plot(x, obj(x, der=der), label='PLSQC', linestyle='--')
    plt.ylim(ylims)
    plt.grid()
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.xlim(x[0], x[-1])
    plt.title(f'Derivative of order {der}')
    plt.savefig(f'img/deriv_order{der}.png', dpi=200)
    