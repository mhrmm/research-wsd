from reed_wsd.analytics import Analytics
from reed_wsd.util import cudaify
from reed_wsd.util import ABS


class Decoder:

    def __call__(self, net, data):
        raise NotImplementedError("This feature needs to be implemented.")


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
        raise NotImplementedError("Must be overridden by inheriting classes.")
    
    def __call__(self, model):
        model = cudaify(model)
        for e in range(self.n_epochs):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            analytics = self.validate_and_analyze(model)
            if self.scheduler is not None:
                self.scheduler.step()
            print("epoch {}:".format(e))
            print("  training loss: ".format(e) + str(batch_loss))
            for key in analytics:
                print('  {}: {}'.format(key, analytics[key]))
        return model, analytics

    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.val_loader, self.trust_model))
        analytics = Analytics(results)
        _, _, auroc = analytics.roc_curve()
        _, _, aupr = analytics.pr_curve()
        _, _, capacity = analytics.risk_coverage_curve()
        avg_err_conf = 0
        avg_crr_conf = 0
        n_error = 0
        n_correct = 0
        n_published = 0
        n_total = len(results)
        for result in results:
            prediction = result['pred']
            gold = result['gold']
            confidence = result['confidence']
            if prediction != ABS:
                n_published += 1
                if prediction == gold:
                    avg_crr_conf += confidence
                    n_correct += 1
                else:
                    avg_err_conf += confidence
                    n_error += 1            
        return {'avg_err_conf': avg_err_conf / n_error if n_error > 0 else 0,
                'avg_crr_conf': avg_crr_conf / n_correct if n_correct > 0 else 0,
                'auroc': auroc,
                'aupr': aupr,
                'capacity': capacity,
                'precision': n_correct / n_published if n_published > 0 else 0,
                'coverage': n_published / n_total if n_total > 0 else 0}
