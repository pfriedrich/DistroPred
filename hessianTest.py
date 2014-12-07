import numpy as np

def hessian ( calculate_cost_function, x0, epsilon=1.e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    from scipy.optimize import approx_fprime
    f1 = approx_fprime( x0, calculate_cost_function, epsilon, *args) 
 
    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1    
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in xrange( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = approx_fprime( x0, calculate_cost_function, epsilon, *args) 
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0        
    return hessian

func=lambda x: x[0]**2+2*x[0]*x[1]+3*x[1]**2+4*x[0]+5*x[1]+6
expected_result=np.zeros((2,2))
expected_result[0][0]=2
expected_result[0][1]=2
expected_result[1][0]=2
expected_result[1][1]=6
result=hessian(func,np.array([0.0,0.0]))
print expected_result
print result

print "------------------------------------------------------"

func=lambda x: x[0]**2+2*x[0]*x[1]+3*x[1]**2+4*x[0]+5*x[1]+6
expected_result=np.zeros((2,2))
expected_result[0][0]=2
expected_result[0][1]=2
expected_result[1][0]=2
expected_result[1][1]=6
result=hessian(func,np.array([1.0,1.0]))
print expected_result
print result

print "------------------------------------------------------"


func2=lambda x: x[0]**3+(x[0]**2)*x[1]-x[1]**2-4*x[1] 
expected_result[0][0]=-4
expected_result[0][1]=0
expected_result[1][0]=0
expected_result[1][1]=-2
result=hessian(func2,np.array([0.0,-2.0]))
print expected_result
print result

print "------------------------------------------------------"

func3=lambda x: x[0]**2+2*x[1]**2+3*x[2]**2+2*x[0]*x[1]+2*x[0]*x[2]+3
expected_result=np.matrix("2 2 2;2 4 0;2 0 6") 
result=hessian(func3,np.array([1.0,1.0,1.0]))
print expected_result
print result

