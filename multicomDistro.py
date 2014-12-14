from optimizer import modelHandler
from optimizer import fitnessFunctions
import numpy as np
import scipy.optimize as sci_opt
from math import exp,fsum,log,cos,pi,sqrt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#this could be replaced when more of optimizer's functionality is used
class dummyOptionObject(object):
    def __init__(self):
        self.spike_thres = 0
        self.output_level = "0"
        self.covariance_flag=1
    def GetUFunString(self):
        return "#\n#\n#\n#\ndef usr_fun(self,v):\n#3\n#cm\n#Ra\n#g_pas\n\tfor sec in h.allsec():\n\t\tsec.cm=v[0]\n\t\tsec.Ra=v[1]\n\t\tfor seg in sec:\n\t\t\tseg.g_pas=v[2]\n\t\t\tseg.e_pas=0"

class simulationEnv(object):
    def __init__(self):
        self.theta_params = []
        self.class_params = []
        self.model_handler = modelHandler.modelHandlerNeuron("/home/fripe/workspace/DistributionPredictor/multi_comp.hoc",".")
        self.dummy_option = dummyOptionObject()
        self.ff = fitnessFunctions.fF(None,self.model_handler,self.dummy_option)
        self.mse = self.ff.calc_ase
        
        self.classes=[(0.1,10)]
        #distros should be given by function, bc checking is needed
        self.theta_distr = [(30,3000),(1e-05,0,001)]
        
        
        
        
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

def drawFromInterval(min, max, sample_num):
    return np.linspace(min, max, sample_num).tolist()

def drawFromUniform(min, max):
    return np.random.uniform(min, max)
 
 
            
def runSimulation(sim,class_sample,theta_sample,run_c_param,args):
    theta_vals=[]
    class_vals=[]
    class_param_prob=[]
    for i in range(theta_sample):
        tmp=[]
        for th in sim.theta_distr:
            tmp.append(drawFromUniform(th[0],th[1]))
        theta_vals.append(tmp)
    
    for i in range(class_sample):
        tmp=[]
        for cl in sim.classes:
            tmp.append(drawFromUniform(cl[0],cl[1]))
        class_vals.append(tmp)
        
    for cl_val in class_vals:
        tmp=[]
        for th_val in theta_vals:
            #sim.setClassParams(sim.class_params,cl_val)
            #sim.setThetaParams(sim.theta_params,th_val)
            param=cl_val
            param.extend(th_val)
            sim.ff.usr_fun(sim.ff,param)
            sim.model_handler.RunControll(run_c_param)
            sim.model_handler.record[0]=downSampleBy(sim.model_handler.record[0],20)#1ms=20 sampling point
            sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{"cov_m":sim.cov_matrix})
            tmp.append(sse)
        #only one class parameter for now
        class_param_prob.append([cl_val[0],fsum(tmp)/theta_sample])

    import scipy.stats
    #works only for 1D normal distro. => only 1 class parameter
    #likelihood*P(D)
    class_param_prob=map(lambda x:[x[0],x[1]*1.0/(sim.classes[0][1]-sim.classes[0][0])],class_param_prob)
    #P(D|R,I)
    class_param_prob=map(lambda x: [x[0],x[1]/fsum(map(lambda x:x[1],class_param_prob))],class_param_prob)
    
    fig4=plt.figure()
    ax4=fig4.add_subplot(111)
    results_to_plot=zip(*class_param_prob)
    ax4.plot(results_to_plot[0],results_to_plot[1],"g.")
    plt.title("Probability distribution of "+sim.class_params[0])
    plt.xlabel("parameter values")
    plt.ylabel("probability")
  
def main():
    sim=simulationEnv()
    sim.model_handler.hoc_obj.psection()
    #sim.defineThetaParams(["soma pas g_pas"],[0.0001])
    #sim.defineClassParams(["soma cm"],[1])
    sim.class_params.append("cm")
    sim.theta_params.append("Ra")
    sim.theta_params.append("g_pas")
    sim.ff.usr_fun(sim.ff,[7.5,151,0.0003])            
    sim.setStimuli(["IClamp",0.5,"soma"], [0.1,30,100])
    sim.model_handler.hoc_obj.psection()
    print "simulation started"
    run_c_param = [199.99,0.01,"v","soma",0.5,-0.0]
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
    
    runSimulation(sim, 20, 10, run_c_param, [])
    plt.show()                
        
if __name__ == "__main__":
    main()

