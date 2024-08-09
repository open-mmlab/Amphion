import torch
import copy

class EMA(object):
    def __init__(self, 
                 model, 
                 decay=0.99, 
                 update_interval=1,
                 device=torch.device('cpu')):

        self.decay = decay
        self.update_iterval = update_interval
        self.device = device

        self.model = model
        with torch.no_grad():
            if hasattr(model, 'get_ema_model') and callable(model.get_ema_model):
                self.ema_model = copy.deepcopy(model.get_ema_model())
                self.cur_state_dict = model.get_ema_model().state_dict()
            else:
                self.ema_model = copy.deepcopy(model)  
                self.cur_state_dict = model.state_dict()
        self.ema_model.to(self.device) 
        self.cur_state_dict = {k: v.clone().to(self.device) for k, v in self.cur_state_dict.items()}

    def update(self, iteration):
        if (iteration + 1) % self.update_iterval == 0:
            # print('{} Update ema'.format(iteration))
            if hasattr(self.model, 'get_ema_model') and callable(self.model.get_ema_model):
                cur_state_dict = self.model.get_ema_model().state_dict()
            else:
                cur_state_dict = self.model.state_dict()

            ema_state_dict = self.ema_model.state_dict()
            for k in ema_state_dict.keys():
                ema_state_dict[k] = ema_state_dict[k] * self.decay + cur_state_dict[k].clone().to(self.device) * (1-self.decay)
            self.ema_model.load_state_dict(ema_state_dict)

    def state_dict(self):
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        state_dict_ = {k: v.clone().to(self.device) for k, v in state_dict.items()}
        self.ema_model.load_state_dict(state_dict_, strict=strict)

    def modify_to_inference(self):
        # get current model
        if hasattr(self.model, 'get_ema_model') and callable(self.model.get_ema_model):
            self.cur_state_dict = self.model.get_ema_model().state_dict()
        else:
            self.cur_state_dict = self.model.state_dict()
        self.cur_state_dict = {k: v.clone().to(self.device) for k, v in self.cur_state_dict.items()}

        ema_state_dict = self.ema_model.state_dict()
        ema_state_dict = {k: v.to(self.model.device) for k, v in ema_state_dict.items()}
        if hasattr(self.model, 'get_ema_model') and callable(self.model.get_ema_model):
            self.model.get_ema_model().load_state_dict(ema_state_dict)
        else:
            self.model.load_state_dict(ema_state_dict)

    def modify_to_train(self):
        self.cur_state_dict = {k: v.clone().to(self.model.device) for k, v in self.cur_state_dict.items()}
        if hasattr(self.model, 'get_ema_model') and callable(self.model.get_ema_model):
            self.model.get_ema_model().load_state_dict(self.cur_state_dict)
        else:
            self.model.load_state_dict(self.cur_state_dict)


