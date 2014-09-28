from neuron import h
from nrn import *




class simulationEnv(object):
    def __init__(self):
        self.theta_params = []
        self.class_params = []
        self.stimulus=None
        self.sections={}
        
    def createOneCompModel(self):
        """
        creates a passive neuron model with one compartment
        """
        #not working with dot notation, but it serves test purposes only
        #later models will be loaded from files
        h('create soma')
        for n in h.allsec():
            self.sections[str(h.secname())]=n
        
    def setMorphParam(self,_p,_v,container):
        if (h.cas().name()==_p[0]):
            h.cas().__setattr__(_p[-1],_v)
            container.append(_p)
        else:
            #change section
            raise NotImplementedError("Only works for one compartment")


    def setChannelParam(self, _p,_v,container):
        raise NotImplementedError("Only works for passive models")
    
    
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

    
    def setStimuli(self, stim_list):
        for stim_object in stim_list:
            self.stimulus=h.IClamp(stim_object.pos_in_section,sec=self.sections[stim_object.section])
            self.stimulus.amp=stim_object.amplitude
            self.stimulus.delay=stim_object.delay
            self.stimulus.dur=stim_object.duration

    
    
    
    

    
    



class Stimulation(object):
    
    def __init__(self, section, postion, stim_type):
        self.section=section
        self.pos_in_section=postion
        self.type=stim_type
        self.delay=None
        self.duration=None
        self.amplitude=None

    
    def setParams(self, delay, duration, amplitude):
        self.delay=delay
        self.duration=duration
        self.amplitude=amplitude

    
    


def main():
    sim=simulationEnv()
    sim.createOneCompModel()
    h.psection()
    sim.setThetaParams(["soma Ra"],[50.0])
    sim.setClassParams(["soma cm"],[0.01])            
    stim=Stimulation("soma",0.5,"IClamp")
    stim.setParams(100,500,0.2)
    sim.setStimuli([stim])
    h.psection()
    print sim.theta_params,sim.class_params
    
if __name__ == "__main__":
    main()