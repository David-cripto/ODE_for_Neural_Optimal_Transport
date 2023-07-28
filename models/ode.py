import torch
import torch.nn as nn
from torchdyn.core import NeuralODE


"""Generic Neural Ordinary Differential Equation.

Args:
    vector_field ([Callable]): the vector field, called with vector_field(t, x) for vector_field(x). 
                               In the second case, the Callable is automatically wrapped for consistency
    solver (Union[str, nn.Module]): 
    order (int, optional): Order of the ODE. Defaults to 1.
    atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
    rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
    sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
    solver_adjoint (Union[str, nn.Module, None], optional): ODE solver for the adjoint. Defaults to None.
    atol_adjoint (float, optional): Defaults to 1e-6.
    rtol_adjoint (float, optional): Defaults to 1e-6.
    integral_loss (Union[Callable, None], optional): Defaults to None.
    seminorm (bool, optional): Whether to use seminorms for adaptive stepping in backsolve adjoints. Defaults to False.
    return_t_eval (bool): Whether to return (t_eval, sol) or only sol. Useful for chaining NeuralODEs in nn.Sequential.
    optimizable_parameters (Union[Iterable, Generator]): parameters to calculate sensitivies for. Defaults to ().
Notes:
    In torchdyn-style, forward calls to a Neural ODE return both a tensor t_eval of time points at which the solution is evaluated
    as well as the solution itself. This behavior can be controlled by setting return_t_eval to False. Calling trajectory also returns
    the solution only. 

    The Neural ODE class automates certain delicate steps that must be done depending on the solver and model used. 
    The prep_odeint method carries out such steps. Neural ODEs wrap ODEProblem.
"""
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc, t_eval=torch.linspace(0,1,6).float(),
                 sensitivity='autograd', method='euler', tol=1e-4):
        super(ODEBlock, self).__init__()    
        self.odefunc = odefunc  
        # euler_eval_nfe=1, dopri5_eval_nfe=36 per interval [0, 1], return [0,1]
        # euler_eval_nfe=9,  , dopri5_eval_nfe=64 per interval [0,1,2...9]
        self.integration_time = t_eval # self.integration_time =  torch.tensor([0,1]).float()  
        self.tol = tol
        # solver can be nn.Module? -> if so, need to think about euler
        self.model = NeuralODE(self.odefunc, sensitivity=sensitivity,  solver=method,
                              rtol = self.tol, atol=self.tol, return_t_eval=True) 
        
    def forward(self, x, return_trajectory=False):
        self.integration_time = self.integration_time.type_as(x)
        t_eval, trajectory = self.model(x, self.integration_time)
        # print('t_eval =', t_eval )
        # print('trajectory shape =', trajectory.shape)
        if return_trajectory:
            return t_eval, trajectory
        else: 
            return trajectory[-1]
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        

class RegDynamics(nn.Module):
    def __init__(self, odefunc):
        super(RegDynamics, self).__init__()    
        self.odefunc = odefunc 
        self.nfe = 0
        
    def forward(self, t, x, args=None): 
        # print('t= ', t)
        self.nfe += 1
        res = self.odefunc(t, x[:x.shape[0]//2])
        # res2 = torch.mean(res.square(), dim = tuple(range(1, len(res.shape))))
        res2 = torch.cat([res, res.square()], dim = 0)
        # print(res.shape)
        # print(res2.shape)
        return res2
    
    
    
    
class ODEBlock2(nn.Module):

    def __init__(self, odefunc, t_eval=torch.linspace(0,1,6).float(),
                 sensitivity='autograd', method='euler', tol=1e-4):
        super(ODEBlock2, self).__init__()    
        self.odefunc = RegDynamics(odefunc)  
        # euler_eval_nfe=1, dopri5_eval_nfe=36 per interval [0, 1], return [0,1]
        # euler_eval_nfe=9,  , dopri5_eval_nfe=64 per interval [0,1,2...9]
        self.integration_time = t_eval # self.integration_time =  torch.tensor([0,1]).float()  
        self.tol = tol
        
        # self.dynamic_function = 
        # solver can be nn.Module? -> if so, need to think about euler
        self.model = NeuralODE(self.odefunc, sensitivity=sensitivity,  solver=method,
                              rtol = self.tol, atol=self.tol, return_t_eval=True) 
        
    def forward(self, x, return_trajectory=False):
        self.integration_time = self.integration_time.type_as(x)
        expand_x = torch.cat((x, torch.zeros(x.shape).type_as(x)), dim = 0)
        t_eval, trajectory = self.model(expand_x, self.integration_time)
        regulizer = trajectory[-1][trajectory.shape[1]//2:]
        trajectory = trajectory[:, :trajectory.shape[1]//2, ...]  
        # print('regul ', regulizer.shape)
        # print('t_eval =', t_eval )
        # print('trajectory shape =', trajectory.shape)
        # trajectory: t b d1 d2 ..
        if return_trajectory:
            return t_eval, trajectory, regulizer.mean()
        else: 
            return trajectory[-1], regulizer.mean()
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value    