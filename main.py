from optimizer import modelHandler
from random import gauss
import numpy as np





class simulationEnv(object):
    def __init__(self):
        self.theta_params = []
        self.class_params = []
        self.model_handler = modelHandler.modelHandlerNeuron("/home/fripe/workspace/DistributionPredictor/one_comp.hoc",".")
        #distros should be given by function, bc checking is needed
        self.theta_distr = (34,5,1)
        #classes has n elements if there are n possible classes
        #each class has m class parameters
        #one class parameter is a tuple: mean of gaussian, std_dev of gaussian
        self.classes=[[(0.2,1)],[(8,1)]]
        
        
        
    def setMorphParam(self,_p,_v,container):
        self.model_handler.SetMorphParameters(_p[0], _p[-1], _v)
        container.append(_p)
        

    def setChannelParam(self, _p,_v,container):
        self.model_handler.SetChannelParameters(_p[0], _p[1], _p[2], _v)
        container.append(_p)
    
    
    def setThetaParams(self,param_list,value_list):
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

    
    def setClassParams(self, param_list, value_list):
        for param,value in zip(param_list,value_list):
            _p=param.split()
            if len(_p)==2:
                self.setMorphParam(_p,value,self.class_params)
            elif len(_p)==3:
                self.setChannelParam(_p,value,self.class_params)
            else:
                raise RuntimeError

    
    def setStimuli(self, stim_creation,stim_param):
        self.model_handler.CreateStimuli(stim_creation)
        self.model_handler.SetStimuli(stim_param, [])

    
    


def main():
    sim=simulationEnv()
    sim.model_handler.hoc_obj.psection()
    sim.setThetaParams(["soma Ra"],[50.0])
    sim.setClassParams(["soma cm"],[0.1])            
    sim.model_handler.hoc_obj.psection()
    sim.setStimuli(["IClamp",0.5,"soma"], [0.2,300,500])
    print "simulation started"
    sim.model_handler.RunControll([1000,0.025,"v","soma",0.5,-65.0])
    sim.base_trace=np.array(sim.model_handler.record)
    print "done simulating"
    print "creating white noise"
    sim.noise_signal=np.random.normal(0,1,len(sim.base_trace))
    sim.signal=np.add(sim.base_trace,sim.noise_signal)
    print "noise added"

    print sim.theta_params,sim.class_params
    
if __name__ == "__main__":
    main()