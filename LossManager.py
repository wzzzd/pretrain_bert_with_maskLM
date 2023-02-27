
from torch.nn import CrossEntropyLoss


class LossManager(object):
    
    def __init__(self, loss_type='ce'):
        self.loss_func = CrossEntropyLoss()

    
    def compute(self, 
                input_x, 
                target,
                hidden_emb_x=None, 
                hidden_emb_y=None, 
                alpha=0.5):
        """        
        计算loss
        Args:
            input: [N, C]
            target: [N, ]
        """
        loss = self.loss_func(input_x, target)
        return loss