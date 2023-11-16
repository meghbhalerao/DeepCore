import torch.optim as optim
import torch
import torch.nn as nn
import os
import time
import logging
logger = logging.getLogger(__name__)
import wandb

class ResNet9Trainer():
    def __init__(self, model, dataloaders, dataset_sizes,  device, num_epochs: int = 100, lr: float = 1e-2, number_of_shuffling: int = 0, save_model_weights: bool = False) -> None:
        self.lr = lr
        self.device = device
        self.num_epochs = num_epochs
        self.model = model
        self.number_of_shuffling = number_of_shuffling
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.save_model_weights = save_model_weights


    def run_train(self):
        print(f"Number of epochs is {self.num_epochs}")
        self.model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
    
        # Set up one-cycle learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, self.lr, epochs=self.num_epochs,steps_per_epoch=len(self.dataloaders["train"]),)
        grad_clip = 0.1
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}/{self.num_epochs - 1}")
            print("-" * 10)
            for phase in ["train", "test"]:
                self.train_or_eval_one_epoch(
                    self.model, criterion, optimizer, phase, self.dataloaders[phase], self.dataset_sizes[phase], epoch, scheduler,grad_clip, self.number_of_shuffling, self.save_model_weights)
                # Step the pseudo random number generator (PRNG) as if we run test phase
                # after every training epoch. Only uncomment this if you are NOT validating
                # the model (i.e., running the test phase) after every epoch.
                # torch.empty((), dtype=torch.int64).random_().item()

        time_elapsed = time.time() - since
        logger.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        phase = "test"
        final_test_loss, final_test_acc = self.train_or_eval_one_epoch(self.model,
            criterion,
            optimizer,
            phase,
            self.dataloaders[phase],
            self.dataset_sizes[phase],
            epoch,
            scheduler,
            grad_clip)
        return self.model
        
    def train_or_eval_one_epoch(self, model, criterion, optimizer, phase, dataloader, dataset_size, epoch, scheduler, grad_clip, number_of_shuffling=0, save_model_weights=False):
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        # Change the shuffling
        if number_of_shuffling > 0:
            for jj in range(number_of_shuffling):
                # By stepping the pseudo random number generator (PRNG) we generate a new
                # shuffling of the samples prior to every epoch.            
                torch.empty((), dtype=torch.int64).random_().item()

        logger.info(f"len(dataloader)-{phase}: {len(dataloader)}")
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward
            # Track history if only in train.
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        wandb.log({"epoch": epoch, f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc})

        # Saving the model weights if specified.
        if save_model_weights and phase == "train":    
            if not os.path.exists("model_weights"):
                os.mkdir("model_weights")
            torch.save(model.state_dict(), f"model_weights/model_weights_all_data_epoch{epoch}.pth",)

        return epoch_loss, epoch_acc
