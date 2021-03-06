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
        return "#\n#\n#\n#\ndef usr_fun(self,v):\n#3\n#cm\n#Ra\n#g_pas\n\tfor sec in h.allsec():\n\t\tsec.cm=v[2]\n\t\tsec.Ra=v[0]\n\t\tfor seg in sec:\n\t\t\tseg.g_pas=v[1]\n\t\t\tseg.e_pas=0"

class simulationEnv(object):
    def __init__(self):
        self.theta_params = []
        self.class_params = []
        self.model_handler = modelHandler.modelHandlerNeuron("/home/fripe/workspace/DistributionPredictor/multi_comp.hoc",".")
        self.dummy_option = dummyOptionObject()
        self.ff = fitnessFunctions.fF(None,self.model_handler,self.dummy_option)
        self.mse = self.ff.calc_ase
        print self.dummy_option.GetUFunString()        
        self.classes=[(150,30)]
        #distros should be given by function, bc checking is needed
        self.theta_distr = [(0.0005,0.0001),(8,2)]
        
        
        
        
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
            
    def generateColoredNoise(self,params):
        self.autocorr_params=params
        self.noise_signal=coloredNoise(self.autocorr_params, len(self.base_trace))
        self.exp_trace = np.add(self.base_trace, self.noise_signal)
        exp_handler=open("/home/fripe/workspace/DistributionPredictor/input_data2.dat","w")
        for l in self.exp_trace:
            exp_handler.write(str(l))
            exp_handler.write("\n")
        exp_handler.close()
        #get autocorrelation
        
        time_scale=np.linspace(0,199.99,1000)
        data=self.autocorr_params[0]*self.autocorr_params[1]*np.exp(-self.autocorr_params[1]*time_scale)
        self.autocorr = data
        self.cov_matrix=getCovMatrix(self.autocorr)
        #self.cov_matrix=getDummyCovMatrix(self.autocorr)
            
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

def drawFromInterval(min, max, num, act):
    return np.linspace(min, max, num).tolist()[act]

def drawFromUniform(min, max):
    return np.random.uniform(min, max)
 
 
            
def runSimulation(sim,class_sample,theta_sample,run_c_param,args):
    sim.ff = fitnessFunctions.fF(None,sim.model_handler,sim.dummy_option)
    sim.mse = sim.ff.calc_ase
    theta_vals=[]
    class_vals=[]
    class_param_prob=[]
    for i in range(theta_sample):
        tmp=[]
        for th in sim.theta_distr:
            tmp.append(drawFromGaussian(th[0],th[1]))
        theta_vals.append(tmp)
    
    for i in range(class_sample):
        tmp=[]
        for cl in sim.classes:
            #tmp.append(drawFromGaussian(cl[0],cl[1]))
            tmp.append(drawFromInterval(100,300,class_sample,i))
        class_vals.append(tmp)
      
    print class_vals
    print theta_vals
    best_fit=1e+10
    iter_counter=0
    for cl_val in class_vals:
        tmp=[]
        print iter_counter
        iter_counter+=1
        for th_val in theta_vals:
            #sim.setClassParams(sim.class_params,cl_val)
            #sim.setThetaParams(sim.theta_params,th_val)
            from copy import copy
            param=copy(cl_val)
            param.extend(th_val)
            print param
            sim.ff.usr_fun(sim.ff,param)
            sim.model_handler.RunControll(run_c_param)
            sim.model_handler.record[0]=downSampleBy(sim.model_handler.record[0],20)#1ms=20 sampling point
            #calculation for colored noise with cov matrix
            sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{"cov_m":sim.cov_matrix})
            #calculation for white noise
#            sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{})
            if sse<best_fit:
                best_fit=sse
            tmp.append(sse)
            print sse
            #tmp.append(exp(-sse))
        #only one class parameter for now
        class_param_prob.append([cl_val[0],tmp])
    
    class_param_prob=map(lambda x:[x[0],map(lambda y:y-best_fit,x[1])],class_param_prob)
    #this belongs to white noise
#    class_param_prob=map(lambda x:[x[0],fsum(map(lambda y:exp(-y/(2*args["noise_params"][1]**2)),x[1]))/theta_sample],class_param_prob)
    #calculation for colored noise
    class_param_prob=map(lambda x:[x[0],fsum(map(lambda y:exp(-y),x[1]))/theta_sample],class_param_prob)
    fig4=plt.figure()
    ax4=fig4.add_subplot(111)
    results_to_plot=zip(*class_param_prob)
    ax4.plot(results_to_plot[0],results_to_plot[1],"r.")
    plt.title("Likelihood of "+sim.class_params[0])
    plt.xlabel("parameter values")
    plt.ylabel("likelihood")
    fig4.savefig("like_ra.svg", dpi=None, facecolor='w', edgecolor='w')
  

    import scipy.stats
    #works only for 1D normal distro. => only 1 class parameter
    #likelihood*P(D)
    class_param_prob=map(lambda x:[x[0],x[1]*scipy.stats.norm(loc=sim.classes[0][0], scale=sim.classes[0][1]).pdf(x[0])],class_param_prob)
    #P(D|R,I)
    class_param_prob=map(lambda x: [x[0],x[1]/fsum(map(lambda x:x[1],class_param_prob))],class_param_prob)
    
    fig5=plt.figure()
    ax5=fig5.add_subplot(111)
    results_to_plot=zip(*class_param_prob)
    ax5.plot(results_to_plot[0],results_to_plot[1],"g.")
    plt.title("Probability distribution of "+sim.class_params[0])
    plt.xlabel("parameter values")
    plt.ylabel("probability")
    fig5.savefig("prob_distro_ra.svg", dpi=None, facecolor='w', edgecolor='w')
  
def main():
    sim=simulationEnv()
    sim.model_handler.hoc_obj.psection()
    #sim.defineThetaParams(["soma pas g_pas"],[0.0001])
    #sim.defineClassParams(["soma cm"],[1])
    sim.class_params.append("Ra")
    sim.theta_params.append("g_pas")
    sim.theta_params.append("cm")
    sim.ff.usr_fun(sim.ff,[151,0.0003,7.5])                  
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
#    noise_dev=1.0
    noise_dev=0.1
#    sim.generateWhiteNoise(noise_mean, noise_dev)

    #first param is the noise amplitude
    #for big noise, use 10.0
    sim.generateColoredNoise([0.1,1.0/30.0])
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

    
#    fig3=plt.figure()
#    ax3=fig3.add_subplot(111)
#    ax3.plot(range(len(sim.autocorr.tolist())),
#             sim.autocorr.tolist())
#    plt.title("autocorrelation")
    print sim.theta_params,sim.class_params
    
    runSimulation(sim, 100, 100, run_c_param, {"noise_params": [noise_mean,noise_dev]})
    #plt.show()
    fig1.savefig("base_trace_ra.svg", dpi=None, facecolor='w', edgecolor='w')
                 
        
if __name__ == "__main__":
    main()


