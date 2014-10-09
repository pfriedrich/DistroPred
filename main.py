from optimizer import modelHandler
from optimizer import fitnessFunctions
import numpy as np
from math import exp,fsum
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
        self.classes=[[(0.2,1)],[(8,1)]]
        #distros should be given by function, bc checking is needed
        self.theta_distr = [[(34.5,1)],[(34.5,1)]]
        
        
        
        
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

    
def drawFromGaussian(mean, std_dev):
    tmp=np.random.normal(mean,std_dev,1)[0]
    while (tmp<0):
        tmp=np.random.normal(mean,std_dev,1)[0]
    print mean,std_dev,tmp 
    return tmp


def main():
    sim=simulationEnv()
    sim.model_handler.hoc_obj.psection()
    sim.defineThetaParams(["soma Ra"],[50.0])
    sim.defineClassParams(["soma cm"],[0.1])            
    sim.setStimuli(["IClamp",0.5,"soma"], [0.2,300,500])
    sim.model_handler.hoc_obj.psection()
    print "simulation started"
    sim.model_handler.RunControll([1000,0.01,"v","soma",0.5,-65.0])
    sim.base_trace=np.array(sim.model_handler.record[0])
    print "done simulating"
    print "creating white noise"
    noise_mean=0.0
    noise_dev=1.0
    sim.noise_signal=np.random.normal(noise_mean,noise_dev,len(sim.base_trace))
    sim.exp_trace=np.add(sim.base_trace,sim.noise_signal)
    print "noise added"
#    plt.plot(range(len(sim.exp_trace.tolist())),sim.exp_trace.tolist())
#    plt.show()
    print sim.theta_params,sim.class_params
    integration_step=200
    iter=0
    classes_result=[[],[]]
    for cl_idx,cl in enumerate(sim.classes):
        while (iter<integration_step):
            for cl_param_idx,cl_param in enumerate(cl):
                sim.setClassParams([sim.class_params[cl_param_idx]],
                                   [drawFromGaussian(cl_param[0], cl_param[1])])
            for th_param_idx,th_param in enumerate(sim.theta_distr[cl_idx]):
                sim.setThetaParams([sim.theta_params[th_param_idx]],
                                   [drawFromGaussian(th_param[0], th_param[1])])
            iter+=1
            #sim.model_handler.hoc_obj.psection()
            sim.model_handler.RunControll([1000,0.01,"v","soma",0.5,-65.0])
            sse = len(sim.exp_trace)*sim.mse(sim.exp_trace,sim.model_handler.record[0],{})
            print -sse/noise_dev**2
            classes_result[cl_idx].append(exp(-sse/noise_dev**2))
        iter = 0
        
    #needs extension to 2+ classes
    cl1_p=fsum(classes_result[0])
    cl2_p=fsum(classes_result[1])
    print classes_result[0]
    print cl1_p
    print classes_result[1]
    print cl2_p
    if (cl1_p>cl2_p):
        print "recording belongs to class 1"
    else:
        print "recording belongs to class 2"

            
                
        
if __name__ == "__main__":
    main()
    
    ()