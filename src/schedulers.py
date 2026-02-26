class LagrangianLambdaScheduler:
    def __init__(self, optimizer, step_size, gamma, init_lr, init_lamda=None):
        self.optimizer = optimizer
        self.step_size = step_size # interval at which change in learning rate occurs
        self.gamma = gamma # multiplication factor for increasing the learning rate
        self.init_lr = [0.04,0.2,0.3,0.5,0.8,1.5,3,100,1000,10000,100000] # base (small) learning rate used initially
        self.init_lamda = init_lamda # starting value for lambda parameters
        self.last_epoch = 1

    def step(self):
        '''
        Use small init_lr for first step_size epochs to refine cached lambdas
        EXCEPT for the following which are set to init_lr * gamma:
        1. For cached lambdas that are < -1 (equivalent to lambda_plus=0.31) and headed down, i.e. param.grad > 0
        2. For cached lambdas that are < -3 (equivalent to lambda_plus=0.05)
        3. Non-cached lambdas i.e. set to init_lamda
        After step_size epochs, increase lr to init_lr * gamma for all lambdas
        Thereafter, increase lr by a factor of gamma for lambdas that are < 0 and headed down every step_size epochs
        '''
        if self.last_epoch == 1:
            for param_group in self.optimizer.param_groups:
                param = param_group['params'][0]
                if not param.requires_grad: # if parameter is frozen, skip it
                    continue
                if param.dim() == 0:
                    if (self.init_lamda is not None and param.data == self.init_lamda) or (param.data < -3.) or (param.data < -1. and param.grad > 0.):
                        param_group['lr'] = self.init_lr[1]
                    param_group['lr'] = self.init_lr[0]
        elif self.last_epoch == self.step_size:
            for param_group in self.optimizer.param_groups:
                param = param_group['params'][0]
                if not param.requires_grad:
                    continue
                param_group['lr'] = self.init_lr[1]
        elif self.last_epoch % self.step_size == 0:
            idx = self.last_epoch // 10
            for param_group in self.optimizer.param_groups:
                param = param_group['params'][0]
                if not param.requires_grad:
                    continue
                if param.dim() == 0 and param.data < 0. and param.grad > 0.:
                    param_group['lr'] = self.init_lr[idx]
        self.last_epoch += 1
