from contextualized_topic_models.models.ctm import CTM
import os
import torch

class CTModel(CTM):
    def __init__(self, model, vocabulary_size, 
                 inference_type='combined', 
                 **kwargs):
        super().__init__(**kwargs, inference_type=inference_type)
        name = f'{model.split("/")[-1]}_vocab-{vocabulary_size}'
        name = f'{name}_bow-{kwargs["bow_size"]}_comp-{kwargs["n_components"]}'
        name = f'{name}_esize-{kwargs["contextual_size"]}'
        name = f'{name}_batch-{kwargs["batch_size"]}'
        name = f'{name}_lr-{kwargs["lr"]}_epochs-{kwargs["num_epochs"]}'
        self.model_name = f'{name}_act-{kwargs["activation"]}'
        
        
    def save(self, models_dir, final=False):
        if final==False:
            pass 
        else:
            if (self.model is not None) and (models_dir is not None):
                model_dir = self._format_file()
                if not os.path.isdir(os.path.join(models_dir, 'model_weights')):
                    os.makedirs(os.path.join(models_dir, 'model_weights'))

                filename = "epoch_{}".format(self.nn_epoch) + '.pth'
                fileloc = os.path.join(models_dir, 'model_weights', filename)
                with open(fileloc, 'wb') as file:
                    torch.save({'state_dict': self.model.state_dict(),
                                'dcue_dict': self.__dict__}, file)