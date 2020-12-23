from reed_wsd.analytics import EvaluationResult, Analytics, EpochResult
from reed_wsd.util import cudaify


class Decoder:

    def __call__(self, net, data):
        raise NotImplementedError("**ABSTRACT METHOD**")


class Trainer:
    
    def __init__(self, criterion, optimizer, train_loader, val_loader,
                 decoder, n_epochs, trustmodel, scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.decoder = decoder
        self.trust_model = trustmodel
        self.scheduler = scheduler

    def _epoch_step(self, model):
        raise NotImplementedError("**ABSTRACT METHOD**")
    
    def __call__(self, model):
        model = cudaify(model)
        epoch_results = []
        for e in range(self.n_epochs):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            if self.scheduler is not None:
                self.scheduler.step()
            eval_result = self.validate_and_analyze(model)
            epoch_results.append(EpochResult(batch_loss, eval_result))
            print("epoch {}:".format(e))
            print("  training loss: ".format(e) + str(batch_loss))
            print(eval_result)
            analytics = Analytics(epoch_results)
            analytics.show_training_dashboard()
        return model, Analytics(epoch_results)

    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.val_loader,
                                    loss_f=self.criterion,
                                    trust_model=self.trust_model))
        validation_loss = self.decoder.get_loss()
        eval_result = EvaluationResult(results, validation_loss)
        return eval_result
