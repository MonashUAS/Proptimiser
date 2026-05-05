import jax.numpy as jnp
import jax.scipy as jscp
import matplotlib.pyplot as plt
import scipy as scp
import numpy as np

def RationalFunctionInterp(x0, y0, P: int, Q: int, verbose:bool = False,
                           extendedChecking:bool = False, method:str = 'lm', 
                           bounds:tuple=(-np.inf, np.inf)):
    """
    Returns the coefficients of a rational function (division of polynomials) 
    with numerator order of N and denominator order of M. N + M must be >= 3
    The form of the polynomial is (A0 + A1*x + ... + AN*x^N)/(1 + B1*x + ... + BM*x^M)
    P & Q specify the numerator and denominator polynomials, refer to https://www.itl.nist.gov/div898/handbook/pmd/section8/pmd812b.htm 
    for help in choosing a P and Q based on the class of function being interpolated
    """

    numargs = P + Q - 1

    """
    RationalWrapper is defined at runtime based on the values of P and Q. Because the final Q value is fixed and 
    appended in the 'rational' function, we accept one fewer arguments than the sum of P and Q
    """
    def RationalWrapper(x, *coeffs):
        return RationalFunction(x, list(coeffs[0:P]), list(coeffs[P:numargs]))
    
    """
    Specify our initial guess as a ones vector of length equal to the number of free parameters
    Use scipy.optimise.curve_fit to determine the coefficient values
    """
    p0 = np.zeros(numargs)

    popt, pcov = scp.optimize.curve_fit(RationalWrapper, x0, y0, p0,
                                        check_finite=extendedChecking, method=method, 
                                        bounds=bounds,maxfev=10000)

    if verbose:
        print("-----> RFI debug <-----")
        
        print("Covariance matrix:")
        print(pcov)
        
        print("Covariance matrix condition (large, eg. >1e6, is bad and implies redundant parameters):")
        print(jnp.linalg.cond(pcov))
        
        print("Optimised parameters:")
        print(popt)

    """
    Return the free parameters (ie, all terms *except* the constant term in the denominator)
    """
    return popt


def RationalFunction(x, p, q):
    """
    The general rational function description.
    p is a list with the polynomial coefficients in the numerator
    q is a list with the polynomial coefficients (except the first one)
    in the denominator
    The zeroth order coefficient of the denominator polynomial is fixed at 1.
    Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
    zeroth order denominator coefficient must comes last.
    """

    func_eval = jnp.polyval(jnp.asarray(p), jnp.asarray(x)) / jnp.polyval(jnp.append(jnp.asarray(q), jnp.array([1.0])), jnp.asarray(x))
    
    return func_eval


"""
Example code for using the optimization and rational function functions to estimate a given set of points
"""
if __name__=="__main__":

    # generate a function, then add noise to the y values
    x = np.array([0.065, 0.095, 0.125, 0.155, 0.185, 0.215, 0.245, 0.275, 0.305, 0.335, 0.365, 0.395, 0.425, 0.455, 0.485])
    y = np.array([[0.00901035],
                [0.00889995],
                [0.00886316],
                [0.01387794],
                [0.01385043],
                [0.01381405],
                [0.01376441],
                [0.01369745],
                [0.01360939],
                [0.01349678],
                [0.01335645],
                [0.01318556],
                [0.0129816 ],
                [0.01274242],
                [0.01322872]])

    # Call the rational function interpolation tool
    popt = RationalFunctionInterp(x, y, 3, 1, verbose=True, extendedChecking=True)
    
    # Plot the results
    plt.plot(x, y, '.', label='data')
    plt.plot(x, RationalFunction(x, list(popt[0:3]), list(popt[3:len(popt)])), label='fit')
    plt.legend()
    plt.show()