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
#        self.theta_distr = [[(0.0001,1e-10)],[(0.0001,1e-10)]]
        self.theta_distr = [[(0.0001,2e-5)],[(0.0001,2e-5)]]
        
        
        
        
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
        data=self.exp_trace
        baseline=np.array(data)
        n = len(baseline)
        variance = baseline.var()
        baseline = baseline-baseline.mean()
        self.autocorr = np.correlate(baseline, baseline, mode = 'full')[-n:]
        self.cov_matrix=getCovMatrix(self.autocorr)
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
 
 
            
def runSimulation(sim,num_iter,run_c_param,args):
    
    integration_step=num_iter
    _iter=0
    classes_result=[[],[]]
    print "start brute force"
    fig0=plt.figure()
    ax0=fig0.add_subplot(111)
    plt.title("Target trace compared to traces generated during integration")
    plt.ylabel('mV')
    plt.xlabel('points')
    for cl_idx,cl in enumerate(sim.classes):
      
        while (_iter<integration_step):
            for cl_param_idx,cl_param in enumerate(cl):
                sim.setClassParams([sim.class_params[cl_param_idx]],
                                   [drawFromGaussian(cl_param[0], cl_param[1])])
            for th_param_idx,th_param in enumerate(sim.theta_distr[cl_idx]):
                sim.setThetaParams([sim.theta_params[th_param_idx]],
                                   [drawFromGaussian(th_param[0], th_param[1])])
            _iter+=1
            sim.model_handler.RunControll(run_c_param)
            sim.model_handler.record[0]=downSampleBy(sim.model_handler.record[0],20)#1ms=20 sampling point
            sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{"cov_m":sim.cov_matrix})
            #sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{})
            classes_result[cl_idx].append(sse)
            if (cl_idx==0):
                ax0.plot(sim.model_handler.record[0])
        _iter = 0
    ax0.plot(range(len(sim.exp_trace.tolist())),
             sim.exp_trace.tolist(), linewidth=2)
    best_fit = min(min(classes_result[0]),min(classes_result[1]))
    classes_result = map(
                         lambda x: map(lambda y: exp(-1*(y-best_fit))
                                       ,x)
                         ,classes_result
                        )
                
            
    classes_prob=[]
    for cl_vals in classes_result:
        print cl_vals
        cl_p=fsum(cl_vals)/float(len(cl_vals))
        print cl_p
        classes_prob.append(cl_p)
    print "likelihoods: "
    for cl_idx,cl_p in enumerate(classes_prob):
        print "\tclass "+str(cl_idx)+".: "+str(cl_p/fsum(classes_prob))
    #plt.show()
    return classes_result
    
  
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
    #noise_dev=1.0
    noise_dev=0.1
    #sim.generateWhiteNoise(noise_mean, noise_dev)
    sim.generateColoredNoise([30.0,1.0/30.0])
    #sim.generateColoredNoise([0.1,1.0/30.0])
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

#    exp_decay=[]
#    for t in range(len(sim.autocorr)):
#        A,K = sim.autocorr_params
#        exp_decay.append(A * np.exp(-K*t))
#    fig2=plt.figure()
#    ax2=fig2.add_subplot(111)
#    ax2.plot(range(len(sim.autocorr.tolist())),
#             sim.autocorr.tolist(),'ro',
#             range(len(exp_decay)),
#             exp_decay,'b-')
#    plt.title("autocorrelation vs exponential decay")
#    fig3=plt.figure()
#    ax3=fig3.add_subplot(111)
#    ax3.plot(range(len(sim.autocorr.tolist())),
#             sim.autocorr.tolist())
#    plt.title("autocorrelation")
    print sim.theta_params,sim.class_params
    
    
    act_results=[]
    num_o_class=2
    class_convergence=[]
    for plot_idx in range(num_o_class):
        current_fig=plt.figure(4+plot_idx)
        class_convergence.append(current_fig.add_subplot(111))
        plt.title("Convergence speed for "+str(plot_idx+1)+" class")
        plt.ylabel('sum of probabilities')
        plt.xlabel('# iteration')
    
    
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    
    styles = markers+[
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

    colors = ('b', 'r', 'c', 'g', 'm', 'y', 'k')
    
    dot2=['go','gs']
    args={}
    args["noise_mean"]=noise_mean
    args["noise_dev"]=noise_dev
    rep=4
    
    all_results=np.ndarray((num_o_class,100,rep))
    for i in range(rep):
        print i
        act_results=runSimulation(sim,100,run_c_param,args)
        for cl_idx,cl_probs in enumerate(act_results):
            for iter_num in range(1,len(cl_probs)+1):
                all_results[cl_idx][iter_num-1][i]=fsum(cl_probs[0:iter_num])/float(iter_num)
                class_convergence[cl_idx].plot(iter_num,
                                               fsum(cl_probs[0:iter_num])/float(iter_num),
                                               linestyle='None',
                                               marker=styles[i % len(styles)],
                                               color=colors[i % len(colors)]
                                               )
    print "plotting deviation"
    convergence_dev=[]
    for plot_idx in range(num_o_class):
        current_fig=plt.figure()
        convergence_dev.append(current_fig.add_subplot(111))
        plt.title("Average convergence speed for "+str(plot_idx+1)+" class")
        plt.ylabel('Average of probabilities')
        plt.xlabel('# iteration')
    for cl_idx in range(len(all_results)):
        for i in range(len(all_results[cl_idx])):
            convergence_dev[cl_idx].errorbar(i,
                                             fsum(all_results[cl_idx][i])/float(len(all_results[cl_idx][i])),
                                             yerr=np.std(all_results[cl_idx][i]),
                                             fmt=dot2[cl_idx])

    plt.show()
                          
                
        
if __name__ == "__main__":
    main()


    
