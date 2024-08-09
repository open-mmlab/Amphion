from torch.nn.utils import clip_grad_norm_


class ClipGradNorm(object):
    def __init__(self, 
                 start_iteration=0, 
                 end_iteration=-1, # if negative, the norm will be always clipped
                 max_norm=0.5):
        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.max_norm = max_norm
    
        self.last_epoch = -1

    def __call__(self, parameters):
        self.last_epoch += 1
        clip = False
        if self.last_epoch >= self.start_iteration:
            clip = True
        if self.end_iteration > 0 and self.last_epoch < self.end_iteration:
            clip = True 
        if clip:
            clip_grad_norm_(parameters, max_norm=self.max_norm)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}
    

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)