# **P**iecewise-Polynomial **L**east-**SQ**uare Fit with **C**ontinuity Constraints
*(PLSQC)*

This Python package provides the PLSQC class which, given an input 1D array corresponding to a sampled signal, can compute a filtered signal and its derivatives.
The algorithm is based on splitting the overall sampled time interval in multiple windows of equal length. On each window, a least-square polynomial fit of chosen degree (default is 3) is computed. Continuity constraint (by default up to degree-1).

## Tutorial
The main script itself contains the below example.
Upon instanciation of the class, the constrained least-square fit is computed.
```python
# instantiate and solve the LSQC problem
obj = PLSQC(x=x, y=y, T=T, continuity=2, deg=3)
```
The obtained object can then be used to evaluate the fit and its derivatives
```python
# Evaluate the filtered signal on the new grid xi
 filtered_signal = obj(xi)
 # Evaluate the filtered second-derivative
 filtered_derivative = obj(xi, der=2)
```

Here is a comparison of the PLSQC result and the Savitzki-Golay filter from Scipy on a simple test case :

![signal and filter](img/filtering.png | width=100)
![first derivative](img/deriv_order0.png | width=100)
![second derivative](img/deriv_order1.png | width=100)
![third derivative](img/deriv_order2.png | width=100)

Improvements for the future:
* Faster linear solution with sparse matrices.
