from optimizer import modelHandler
from optimizer import fitnessFunctions
import numpy as np
import scipy.optimize as sci_opt
from math import exp,fsum,log,cos,pi,sqrt
import matplotlib.pyplot as plt


class dummyOptionObject(object):
    def __init__(self):
        self.spike_thres = 0
        self.output_level = "0"
        self.covariance_flag=1
    def GetUFunString(self):
        return ""

class simulationEnv(object):
    def __init__(self):
        self.theta_params = []
        self.class_params = []
        self.model_handler = modelHandler.modelHandlerNeuron("/home/fripe/workspace/DistributionPredictor/one_comp.hoc",".")
        self.mse = fitnessFunctions.fF(None,None,dummyOptionObject()).calc_ase
        #classes has n elements if there are n possible classes
        #each class has m class parameters
        #one class parameter is a tuple: mean of gaussian, std_dev of gaussian
        self.classes=[[(0.9,0.1)],[(2,0.1)]]
        #distros should be given by function, bc checking is needed
        self.theta_distr = [[(0.0001,1e-10)],[(0.0001,1e-10)]]
        
        
        
        
    def setMorphParam(self,_p,_v,container):
        self.model_handler.SetMorphParameters(_p[0], _p[-1], _v)
        if container!=None:
            container.append(" ".join(_p))
        

    def setChannelParam(self, _p,_v,container):
        self.model_handler.SetChannelParameters(_p[0], _p[1], _p[2], _v)
        if container!=None:
            container.append(" ".join(_p))
    
    
    def defineThetaParams(self,param_list,value_list):
        """
        Sets the theta parameters to their given values and stores them in the
        simulation environment.
        param_list: list of strings: compartment param, or comparment channel param
        value_list: list of floating values
        """
        for param,value in zip(param_list,value_list):
            _p=param.split()
            if len(_p)==2:
                self.setMorphParam(_p,value,self.theta_params)
            elif len(_p)==3:
                self.setChannelParam(_p,value,self.theta_params)
            else:
                raise RuntimeError

    
    def defineClassParams(self, param_list, value_list):
        for param,value in zip(param_list,value_list):
            _p=param.split()
            if len(_p)==2:
                self.setMorphParam(_p,value,self.class_params)
            elif len(_p)==3:
                self.setChannelParam(_p,value,self.class_params)
            else:
                raise RuntimeError

    def setThetaParams(self,param_list,value_list):
        """
        Sets the theta parameters to their given values and stores them in the
        simulation environment.
        param_list: list of strings: compartment param, or comparment channel param
        value_list: list of floating values
        """
        for param,value in zip(param_list,value_list):
            _p=param.split()
            if (not param in self.theta_params):
                raise RuntimeError(param)
            if len(_p)==2:
                self.setMorphParam(_p,value,None)
            elif len(_p)==3:
                self.setChannelParam(_p,value,None)
            else:
                raise RuntimeError

    
    def setClassParams(self, param_list, value_list):
        for param,value in zip(param_list,value_list):
            _p=param.split()
            if (not param in self.class_params):
                raise RuntimeError(param)
            if len(_p)==2:
                self.setMorphParam(_p,value,None)
            elif len(_p)==3:
                self.setChannelParam(_p,value,None)
            else:
                raise RuntimeError
            
    def setStimuli(self, stim_creation,stim_param):
        self.model_handler.CreateStimuli(stim_creation)
        self.model_handler.SetStimuli(stim_param, [])
        
    def generateWhiteNoise(self, noise_mean, noise_dev):
        self.noise_signal = np.random.normal(noise_mean, noise_dev, len(self.base_trace))
        self.exp_trace = np.add(self.base_trace, self.noise_signal)
            
    def generateColoredNoise(self,source):
        #get experimental data
        f_h=open(source,"r")
        data=[]
        for i in range(4000):
            data.append(float(f_h.readline().split("\t")[-1]))
        f_h.close()
        #get autocorrelation
        baseline=np.array(data)
        n = len(baseline)
        variance = baseline.var()
        baseline = baseline-baseline.mean()
        self.autocorr = np.correlate(baseline, baseline, mode = 'full')[-n:]
        #self.autocorr = r/(variance*(np.arange(n, 0, -1)))
        #fit exponential decay
        def exp_decay(t, A, K):
            return A * np.exp(-K*t)
        params,param_cov=sci_opt.curve_fit(exp_decay, np.array(range(4000)), self.autocorr)
        print params
        #self.autocorr_params=params
        self.autocorr_params=[0.1,1.0/30.0]
        self.noise_signal=coloredNoise(self.autocorr_params, len(self.base_trace))
        self.exp_trace = np.add(self.base_trace, self.noise_signal)
        exp_handler=open("/home/fripe/workspace/DistributionPredictor/input_data2.dat","w")
        for l in self.exp_trace:
            exp_handler.write(str(l))
            exp_handler.write("\n")
        exp_handler.close()
        self.cov_matrix=getCovMatrix(downSampleBy(self.autocorr,4))
        #self.cov_matrix=getDummyCovMatrix(downSampleBy(self.autocorr,4))
        
def getCovMatrix(autocorr):
    n=len(autocorr)
    print "cov dim", n
    tmp=np.zeros((n,n))
    for r in range(n):
        for c in range(r,n):
            tmp[r][r:n]=autocorr[0:n-r]

    result=np.matrix(tmp + tmp.T - np.diag(tmp.diagonal())).I
    cov_handler = open("/home/fripe/workspace/DistributionPredictor/cov_m.dat","w")
    for row in range(n):
        r_str=" ".join(map(str,result[row]))
        r_str=r_str.strip("[")
        r_str=r_str.strip("]")
        if row<n-1:
            r_str=r_str+";\n"
        cov_handler.write(r_str)
        
    cov_handler.close()
                  
    return result

def getDummyCovMatrix(autocorr):
    n=len(autocorr)
    print "cov dim", n
    tmp=np.identity(n)
    return tmp
    
def coloredNoise(params,length):
    noise=[]
    D=params[0]
    lam=params[1]
    delta_t=1
    n,m = np.random.uniform(0.0,1.0,2)
    E=lambda x:exp(-x*delta_t)
    e_prev=sqrt(-2*D*lam*log(m))*cos(2*pi*n)
    noise.append(e_prev)
    for i in range(length-1):
        a,b = np.random.uniform(0.0,1.0,2)
        h=sqrt(-2*D*lam*(1-E(lam)**2)*log(a))*cos(2*pi*b)
        e_next=e_prev*E(lam)+h
        noise.append(e_next)
        e_prev=e_next
    return np.array(noise)
    
def downSampleBy(signal,factor):
    tmp_mod=[]
    for idx in range(0,len(signal),factor):
        tmp_mod.append(signal[idx])
    return tmp_mod    

    
def drawFromGaussian(mean, std_dev):
    tmp=np.random.normal(mean,std_dev,1)[0]
    while (tmp<0):
        tmp=np.random.normal(mean,std_dev,1)[0]
    #print mean,std_dev,tmp 
    return tmp


def runOptimizer(sim):
    import sys
    from optimizer import Core
    import xml.etree.ElementTree as ET
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.interactive(False)
    fname="/home/fripe/workspace/DistributionPredictor/optimizer_config.xml"
    param = None
    try:
        f = open(fname, "r")
    except IOError as ioe:
        print ioe
        sys.exit("File not found!\n")
    tree = ET.parse(fname)
    root = tree.getroot()
    if root.tag != "settings":
        sys.exit("Missing \"settings\" tag in xml!")

    core = Core.coreModul()
    if param != None:
        core.option_handler.output_level = param.lstrip("-v_level=") 
    core.option_handler.read_all(root)
    core.Print()
    kwargs = {"file" : core.option_handler.GetFileOption(),
            "input": core.option_handler.GetInputOptions()}
    core.FirstStep(kwargs)
    
    kwargs = {"simulator": core.option_handler.GetSimParam()[0],
            "model" : core.option_handler.GetModelOptions(),
            "sim_command":core.option_handler.GetSimParam()[1]}
    core.LoadModel(kwargs)
    
    kwargs = {"stim" : core.option_handler.GetModelStim(), "stimparam" : core.option_handler.GetModelStimParam()}
    core.SecondStep(kwargs)
    
    
    kwargs = None
    
    core.ThirdStep(kwargs)
    core.FourthStep()
    print core.optimizer.final_pop[0].candidate[0:len(core.optimizer.final_pop[0].candidate) / 2]
    print "resulting parameters: ", core.optimal_params
    
    normalized_optimum=core.optimizer.final_pop[0].candidate[0:len(core.option_handler.adjusted_params)]
#    import numdifftools as nd
#    hessianObj=nd.Hessian(,numTerms=0)
#    print hessianObj.hessian(normalized_optimum)
    func=lambda x: core.optimizer.ffun([x],{})[0]
    print hessian(func, np.array(normalized_optimum))
 
 
def hessian ( calculate_cost_function, x0, epsilon=1.e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    import numpy as np
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
   
def main():
    sim=simulationEnv()
    sim.model_handler.hoc_obj.psection()
    sim.defineThetaParams(["soma pas g_pas"],[0.0001])
    sim.defineClassParams(["soma cm"],[1])            
    sim.setStimuli(["IClamp",0.5,"soma"], [0.1,30,100])
    sim.model_handler.hoc_obj.psection()
    print "simulation started"
    run_c_param = [199.99,0.01,"v","soma",0.5,-70.0]
    sim.model_handler.RunControll(run_c_param)
    sim.base_trace=np.array(sim.model_handler.record[0])
    sim.base_trace=downSampleBy(sim.base_trace, 20)
    print "done simulating"
    print "creating noise"
    noise_mean=0.0
    noise_dev=1.0
    #sim.generateWhiteNoise(noise_mean, noise_dev)
    sim.generateColoredNoise("/home/fripe/workspace/git/optimizer/tests/ca1pc_anat/131117-C2_short.dat")
    print "noise added"
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    ax1.plot(range(len(sim.exp_trace.tolist())),
             sim.exp_trace.tolist(),
             range(len(sim.exp_trace.tolist())),
             sim.base_trace)
    plt.title("Base trace with noise added")
    plt.ylabel('mV')
    plt.xlabel('points')

    exp_decay=[]
    for t in range(len(sim.autocorr)):
        A,K = sim.autocorr_params
        exp_decay.append(A * np.exp(-K*t))
    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    ax2.plot(range(len(sim.autocorr.tolist())),
             sim.autocorr.tolist(),'ro',
             range(len(exp_decay)),
             exp_decay,'b-')
    plt.title("autocorrelation vs exponential decay")
    fig3=plt.figure()
    ax3=fig3.add_subplot(111)
    ax3.plot(range(len(sim.autocorr.tolist())),
             sim.autocorr.tolist())
    plt.title("autocorrelation")
    print sim.theta_params,sim.class_params
    
    runOptimizer(sim)
    
    
if __name__ == "__main__":
    main()
    