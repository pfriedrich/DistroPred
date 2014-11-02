from optimizer import modelHandler
from optimizer import fitnessFunctions
import numpy as np
import scipy.optimize as sci_opt
from math import exp,fsum,log,cos,pi,sqrt
import matplotlib.pyplot as plt


#this could be replaced when more of optimizer's functionality is used
class dummyOptionObject(object):
    def __init__(self):
        self.spike_thres = 0
        self.output_level = "0"
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
    
    def coloredNoise(self,params,length):
        noise=[]
        lam=params[1]
        D=1
        delta_t=1
        a,b,n,m = np.random.uniform(0.0,1.0,4)
        E=lambda x:exp(-x*delta_t)
        h=sqrt(-2*D*lam*(1-E(lam)**2)*log(a))*cos(2*pi*b)
        e_prev=sqrt(-2*D*lam*log(m))*cos(2*pi*n)
        noise.append(e_prev)
        for i in range(length-1):
            e_next=e_prev*E(lam)+h
            noise.append(e_next)
            e_prev=e_next
        return np.array(noise)
        
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
        r = np.correlate(baseline, baseline, mode = 'full')[-n:]
        assert np.allclose(r, np.array([(baseline[:n-k]*baseline[-(n-k):]).sum() for k in range(n)]))
        normalized_autocorr = r/(variance*(np.arange(n, 0, -1)))
        #fit exponential decay
        def exp_decay(t, A, K):
            return A * np.exp(-t/K)
        params,param_cov=sci_opt.curve_fit(exp_decay, np.array(range(4000)), normalized_autocorr)
        print params
        self.noise_signal=self.coloredNoise(params, len(self.base_trace))
        self.exp_trace = np.add(self.base_trace, self.noise_signal)
        
        
def downSampleBy(signal,factor):
    tmp_mod=[]
    for idx in range(0,len(signal),factor):
        tmp_mod.append(signal[idx])
    print len(tmp_mod)
    return tmp_mod    

    
def drawFromGaussian(mean, std_dev):
    tmp=np.random.normal(mean,std_dev,1)[0]
    while (tmp<0):
        tmp=np.random.normal(mean,std_dev,1)[0]
    #print mean,std_dev,tmp 
    return tmp


def runSimulation(num_iter):
    sim=simulationEnv()
    sim.model_handler.hoc_obj.psection()
    sim.defineThetaParams(["soma pas g_pas"],[0.0001])
    sim.defineClassParams(["soma cm"],[1])            
    sim.setStimuli(["IClamp",0.5,"soma"], [0.1,30,100])
    sim.model_handler.hoc_obj.psection()
    print "simulation started"
    run_c_param = [200,0.01,"v","soma",0.5,-70.0]
    sim.model_handler.RunControll(run_c_param)
    sim.base_trace=np.array(sim.model_handler.record[0])
    sim.base_trace=downSampleBy(sim.base_trace, 20)
    print "done simulating"
    print "creating white noise"
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
    print sim.theta_params,sim.class_params
    integration_step=num_iter
    _iter=0
    classes_result=[[],[]]
    trace_plot_axes=[]
    print "start brute force"
    for cl_idx,cl in enumerate(sim.classes):
#        current_fig=plt.figure()
#        trace_plot_axes.append(current_fig.add_subplot(111))
#        trace_plot_axes[cl_idx].plot(range(len(sim.exp_trace.tolist())),
#                                     sim.exp_trace.tolist(),range(len(sim.exp_trace.tolist())),sim.base_trace)
#        plt.title("Traces for "+str(cl_idx+1)+" class")
#        plt.ylabel('mV')
#        plt.xlabel('points')
        while (_iter<integration_step):
            for cl_param_idx,cl_param in enumerate(cl):
                sim.setClassParams([sim.class_params[cl_param_idx]],
                                   [drawFromGaussian(cl_param[0], cl_param[1])])
            for th_param_idx,th_param in enumerate(sim.theta_distr[cl_idx]):
                sim.setThetaParams([sim.theta_params[th_param_idx]],
                                   [drawFromGaussian(th_param[0], th_param[1])])
            _iter+=1
            #sim.model_handler.hoc_obj.psection()
            sim.model_handler.RunControll(run_c_param)
            sim.model_handler.record[0]=downSampleBy(sim.model_handler.record[0],20)#1ms=20 sampling point
#            trace_plot_axes[cl_idx].plot(range(len(sim.exp_trace.tolist())),
#                                         sim.model_handler.record[0])
            print len(sim.exp_trace.tolist()),len(sim.model_handler.record[0])
            sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{})
            classes_result[cl_idx].append(sse)
            #classes_result[cl_idx].append(exp(-sse/noise_dev**2))
            print _iter
        _iter = 0
    
    best_fit = min(min(classes_result[0]),min(classes_result[1]))
    classes_result = map(
                         lambda x: map(lambda y: exp(-1*(y-best_fit)/(2*noise_dev**2))
                                       ,x)
                         ,classes_result
                        )
                
            
    classes_prob=[]
    for cl_vals in classes_result:
        print cl_vals
        cl_p=fsum(cl_vals)/float(len(cl_vals))
        print cl_p
        classes_prob.append(cl_p)
    print "likelyhoods: "
    for cl_idx,cl_p in enumerate(classes_prob):
        print "\tclass "+str(cl_idx)+".: "+str(cl_p/fsum(classes_prob))
    #plt.show()
    return classes_result
    
  
def main():
    act_results=[]
    num_o_class=2
    class_convergence=[]
    for plot_idx in range(num_o_class):
        current_fig=plt.figure(plot_idx)
        class_convergence.append(current_fig.add_subplot(111))
        plt.title("Convergence speed for "+str(plot_idx+1)+" class")
        plt.ylabel('sum of probabilities')
        plt.xlabel('# iteration')
    
    dot=['ro','bs']
    for i in range(2):
        act_results=runSimulation(100)
        for cl_idx,cl_probs in enumerate(act_results):
            for iter_num in range(1,len(cl_probs)+1):
                class_convergence[cl_idx].plot(iter_num,
                                               fsum(cl_probs[0:iter_num])/float(iter_num),
                                               dot[i])
            
                
    plt.show()
                          
                
        
if __name__ == "__main__":
    main()


    