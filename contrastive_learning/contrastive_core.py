from tqdm import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from dl_training.core import Base


class ContrastiveBase(Base):
    def get_output_pairs(self, inputs):
        """
        :param inputs: torch.Tensor
        :return: pair (z_i, z_j) where z_i and z_j have the same structure as inputs
        """
        z_i = self.model(inputs[:, 0, :].to(self.device))
        z_j = self.model(inputs[:, 1, :].to(self.device))
        return z_i, z_j

    def update_metrics(self, values, nb_batch, logits=None, target=None, validation=False, **kwargs):
        if logits is not None and target is not None:
            for name, metric in self.metrics.items():
                if validation:
                    name = name + " on validation set"
                if name not in values:
                    values[name] = 0
                values[name] += float(metric(logits, target)) / nb_batch

    def train(self, loader, run=None, epoch=None, **kwargs):
        """ Train the model on the dataloader provided
        Parameters
        ----------
        loader: a pytorch Dataloader
        epoch : number of the epoch
        run : number of the run

        Returns
        -------
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        scaler = kwargs.get("gradscaler")
        self.model.train()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc=f"Mini-Batch ({run},{epoch})")

        values = {}
        losses = []
        for dataitem in loader:
            pbar.update()
            inputs = dataitem.inputs
            labels = dataitem.labels.to(self.device) if dataitem.labels is not None else None
            self.optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    (z_i, z_j) = self.get_output_pairs(inputs)
                    if labels is not None:
                        batch_loss, *args = self.loss(z_i, z_j, labels)
                    else:
                        batch_loss, *args = self.loss(z_i, z_j)
                scaler.scale(batch_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                (z_i, z_j) = self.get_output_pairs(inputs)
                if labels is not None:
                    batch_loss, *args = self.loss(z_i, z_j, labels)
                else:
                    batch_loss, *args = self.loss(z_i, z_j)

                batch_loss.backward()
                self.optimizer.step()

            aux_losses = (self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
            for name, aux_loss in aux_losses.items():
                if name not in values:
                    values[name] = 0
                values[name] += float(aux_loss) / nb_batch

            losses.append(float(batch_loss))
            self.update_metrics(values, nb_batch, *args, **kwargs)

        loss = np.mean(losses)

        pbar.close()
        return loss, values

    def test(self, loader, set_name="validation", with_visuals=False, **kwargs):
        """ Evaluate the model on the validation data. The tests is done in a usual way for a supervised task.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.

        Returns
        -------
        y: array-like
            the predicted data.
        y_true: array-like
            the true data
        X: array_like
            the input data
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """

        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc=f"Mini-Batch")
        loss = 0
        values = {}
        visuals = []
        y, y_true, X = [], [], []

        with torch.no_grad():
            for dataitem in loader:
                pbar.update()
                inputs = dataitem.inputs
                labels = dataitem.labels.to(self.device) if dataitem.labels is not None else None

                (z_i, z_j) = self.get_output_pairs(inputs)
                if with_visuals:
                    visuals.append(self.model.get_current_visuals())

                if labels is not None:
                    batch_loss, *args = self.loss(z_i, z_j, labels)
                else:
                    batch_loss, *args = self.loss(z_i, z_j)

                loss += float(batch_loss) / nb_batch
                #y.extend(logits.detach().cpu().numpy())
                #y_true.extend(target.detach().cpu().numpy())

                # eventually appends the inputs to X
                #for i in inputs:
                #    X.extend(i.cpu().detach().numpy())

                # Now computes the metrics with (y, y_true)
                self.update_metrics(values, nb_batch, *args, validation=True, **kwargs)

                aux_losses = (self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
                for name, aux_loss in aux_losses.items():
                    name += f" on {set_name} set"
                    if name not in values:
                        values[name] = 0
                    values[name] += aux_loss / nb_batch

        pbar.close()

        if len(visuals) > 0:
            visuals = np.concatenate(visuals, axis=0)

        if with_visuals:
            return y, y_true, X, loss, values, visuals

        return y, y_true, X, loss, values