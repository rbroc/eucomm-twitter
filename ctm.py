from contextualized_topic_models.models.ctm import CombinedTM


class CTM(CombinedTM):
    def __init__(self, vocabulary_size, **kwargs):
        super().__init__(**kwargs)
        name = f'{kwargs["model"].split("/")[-1]}'
        name = f'{name}_vocab-{kwargs["vocabulary_size"]}'
        name = f'{name}_bow-{kwargs["bow_size"]}_comp-{kwargs["n_components"]}'
        name = f'{name}_esize-{kwargs["contextual_size"]}'
        name = f'{name}_batch-{kwargs["batch_size"]}'
        name = f'{name}_lr-{kwargs["lr"]}_epochs-{kwargs["num_epochs"]}'
        self.model_name = f'{name}_act-{kwargs["activation"]}'
        
        
    def save(self, models_dir):
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, self.model_name)):
                os.makedirs(os.path.join(models_dir, self.model_name))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, self.model_name, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)