from .Model_Trainer import Model_Trainer
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from tqdm import tqdm
import Model.VisualTransformer as VisualTransformer
import Model.ml_metrics_utils as ml_utils
import numpy as np
from torch.utils.data import DataLoader
import multiprocessing
from sklearn.metrics import confusion_matrix, classification_report
import wandb

class CNN_Trainer(Model_Trainer):
    """
    Training class for CNN based (resnet, convnext_tiny_custom, vit_tiny_patch) NOT (unlabeled_domain or in MLP, SVC, RF).
    """

    def __init__(self, X_data, Y_data, label_data, env):
        super().__init__(X_data, Y_data, label_data, env)

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def model_loop(self):
        self.pretrain_model() 
        if self.args.pretrain_and_finetune:
            self.finetune_model()
        
    def setup_model(self):
        """
        Set up all model parameters for the run. 
        """
        super().set_pretrain_path()
        self.set_model()
        self.set_optimizer()
        self.set_param_requires_grad()
        super().set_resize_transform()
        super().set_loaders()
        super().set_criterion()
        super().start_pretrain_run()
        super().set_model_to_device()
        super().set_testrun_foldername()
        super().set_gesture_labels()
        super().plot_images()

    def set_model(self):

        if self.args.model == 'resnet50_custom':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(self.model.children())[:-4])
            # #self.model = nn.Sequential(*list(self.model.children())[:-4])
            num_features = self.model[-1][-1].conv3.out_channels
            # #num_features = self.model.fc.in_features
            dropout = 0.5
            self.model.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
            self.model.add_module('fc1', nn.Linear(num_features, 512))
            self.model.add_module('relu', nn.ReLU())
            self.model.add_module('dropout1', nn.Dropout(dropout))
            self.model.add_module('fc3', nn.Linear(512, self.num_gestures))
            self.model.add_module('softmax', nn.Softmax(dim=1))

        elif self.args.model == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Replace the last fully connected layer
            num_ftrs = self.model.fc.in_features  # Get the number of input features of the original fc layer
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)  # Replace with a new linear layer

        elif self.args.model == 'convnext_tiny_custom':
            class LayerNorm2d(nn.LayerNorm):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = x.permute(0, 2, 3, 1)
                    x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                    x = x.permute(0, 3, 1, 2)
                    return x
                
            # Referencing: https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098

            n_inputs = 768
            hidden_size = 128 # default is 2048
            n_outputs = self.num_classes

            # self.model = timm.create_model(self.model_name, pretrained=True, num_classes=10)
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            #self.model = nn.Sequential(*list(self.model.children())[:-4])
            #self.model = nn.Sequential(*list(self.model.children())[:-3])
            #num_features = self.model[-1][-1].conv3.out_channels
            #num_features = self.model.fc.in_features
            dropout = 0.1 # was 0.5

            sequential_layers = nn.Sequential(
                LayerNorm2d((n_inputs,), eps=1e-06, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(n_inputs, hidden_size, bias=True),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, n_outputs),
                nn.LogSoftmax(dim=1)
            )
            self.model.classifier = sequential_layers

        elif self.args.model == 'vit_tiny_patch2_32':
            pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
            self.model = VisualTransformer.vit_tiny_patch2_32(pretrained=True, pretrained_path=pretrain_path, num_classes=self.num_classes)

        elif self.args.model not in ["MLP", "SVC", "RF"]:
            if self.args.force_regression: 
                self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.Y.validation.shape[1])
            else: 

                self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.Y.validation.shape[1])
                
        # Calculate the number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        # Print the total number of parameters
        print(f'Total number of parameters: {total_params}')

    def set_param_requires_grad(self):
        
        num = 0
        for name, param in self.model.named_parameters():
            num += 1
            if (num > 0):
            #if (num > 72): # for -3
            #if (num > 33): # for -4
                param.requires_grad = True
            else:
                param.requires_grad = False

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def print_classification_metrics(self):
        """
        Batches data for test, train, and validation and plots confusion matrix and classification report.
        """

        # Test Batch
        self.model.eval()
        with torch.no_grad():
            test_predictions = []
            for X_batch, Y_batch in tqdm(self.test_loader, desc="Test Batch Loading for Confusion Matrix"):
                X_batch = X_batch.to(self.device).to(torch.float32)
                outputs = self.model(X_batch)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                test_predictions.extend(preds)

        true_labels = np.argmax(self.Y.test.cpu().detach().numpy(), axis=1)
        test_predictions = np.array(test_predictions)

        # Calculate and print the confusion matrix
        conf_matrix = confusion_matrix(true_labels, test_predictions)
        print("Confusion Matrix:")
        print(conf_matrix)

        print("Classification Report:")
        print(classification_report(true_labels, test_predictions))
        
        self.utils.plot_confusion_matrix(np.argmax(self.Y.test.cpu().detach().numpy(), axis=1), np.array(test_predictions), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'test')  

        torch.cuda.empty_cache()  # Clear cache if needed 

        # Validation Metrics
        self.model.eval()
        with torch.no_grad():
            validation_predictions = []
            for X_batch, Y_batch in tqdm(self.val_loader, desc="Validation Batch Loading for Confusion Matrix"):
                X_batch = X_batch.to(self.device).to(torch.float32)
                outputs = self.model(X_batch)
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                preds = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                validation_predictions.extend(preds)
        
        self.utils.plot_confusion_matrix(np.argmax(self.Y.validation.cpu().detach().numpy(), axis=1), np.array(validation_predictions), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'validation')   
        torch.cuda.empty_cache()

        # Load training in smaller batches for memory purposes
        torch.cuda.empty_cache()  # Clear cache if needed

        self.model.eval()
        self.train_loader_unshuffled = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=self.utils.seed_worker, pin_memory=True, drop_last=self.args.force_regression)
        
        # Train Metrics
        with torch.no_grad():
            train_predictions = []
            for X_batch, Y_batch in tqdm(self.train_loader_unshuffled, desc="Training Batch Loading for Confusion Matrix"):
                X_batch = X_batch.to(self.device).to(torch.float32)
                outputs = self.model(X_batch)
                if isinstance(outputs, dict):
                        outputs = outputs['logits']
                preds = torch.argmax(outputs, dim=1)
                train_predictions.extend(preds.cpu().detach().numpy())
        
        self.utils.plot_confusion_matrix(np.argmax(self.Y.train.cpu().detach().numpy(), axis=1), np.array(train_predictions), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'train')
    
    def finetune_model(self):
        """
        Start a new loop for finetuning. 
        """

        self.ft_run = wandb.init(name=self.wandb_runname+"_finetune", project=self.project_name) 
        ft_epochs = self.args.finetuning_epochs
        finetune_dataset = super().CustomDataset(self.X.train_finetuning,self.Y.train_finetuning, transform=self.resize_transform)
        finetune_loader = DataLoader(finetune_dataset, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//8, worker_init_fn=self.utils.seed_worker, pin_memory=True, drop_last=self.args.force_regression)

        # Initialize metrics for finetuning training and validation
        ft_training_metrics, ft_validation_metrics, ft_testing_metrics = super().get_metrics()

        # Finetuning Loop 
        for epoch in tqdm(range(ft_epochs), desc="Finetuning Epoch"):
            self.model.train()
            train_loss = 0.0
            
            for ft_train_metric in ft_training_metrics:
                ft_train_metric.reset()

            with tqdm(finetune_loader, desc=f"Finetuning Epoch {epoch+1}/{ft_epochs}", leave=False) as t:
                for X_batch, Y_batch in t:
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch =Y_batch.to(self.device).to(torch.float32)
                    if self.args.force_regression:
                        Y_batch_long =Y_batch
                    else: 
                        Y_batch_long = torch.argmax(Y_batch, dim=1)

                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    loss = self.criterion(output,Y_batch_long)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    for ft_train_metric in ft_training_metrics:
                        ft_train_metric(output,Y_batch_long)

                    if not self.args.force_regression: 
                        if t.n % 10 == 0:
                            ft_accuracy_metric = next(metric for metric in ft_training_metrics if metric.name =="Micro_Accuracy")

                            t.set_postfix({
                                "Batch Loss": loss.item(), 
                                "Batch Acc": ft_accuracy_metric.compute().item()
                            })

            # Finetuning Validation
            self.model.eval()
            val_loss = 0.0

            for ft_val_metric in ft_validation_metrics:
                ft_val_metric.reset()

            if not self.args.force_regression:
                all_val_outputs = []
                all_val_labels = []

            with torch.no_grad():
                for X_batch, Y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32)
                    if self.args.force_regression:
                        Y_batch_long =Y_batch
                    else: 
                        Y_batch_long = torch.argmax(Y_batch, dim=1)

                    output = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    val_loss += self.criterion(output,Y_batch).item()

                    for ft_val_metric in ft_validation_metrics:
                        if ft_val_metric.name != "Macro_AUROC" and ft_val_metric.name != "Macro_AUPRC":
                            ft_val_metric(output,Y_batch_long)

                    if not self.args.force_regression:
                        all_val_outputs.append(output)
                        all_val_labels.append(Y_batch_long)

            if not self.args.force_regression:
                all_val_outputs = torch.cat(all_val_outputs, dim=0)
                all_val_labels = torch.cat(all_val_labels, dim=0)  
                Y_validation_long = torch.argmax(self.Y.validation, dim=1).to(self.device)

                true_labels =Y_validation_long.cpu().detach().numpy()
                test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)
                conf_matrix = confusion_matrix(true_labels, test_predictions)
                print("Confusion Matrix:")
                print(conf_matrix)

                finetune_val_macro_auroc_metric = next(metric for metric in ft_validation_metrics if metric.name == "Macro_AUROC")
                finetune_val_macro_auprc_metric = next(metric for metric in ft_validation_metrics if metric.name == "Macro_AUPRC")

                finetune_val_macro_auroc_metric(all_val_outputs,Y_validation_long)
                finetune_val_macro_auprc_metric(all_val_outputs,Y_validation_long)

            # Calculate average loss and metrics
            train_loss /= len(finetune_loader)
            val_loss /= len(self.val_loader)
            if not self.args.force_regression: 
                tpr_results = ml_utils.evaluate_model_tpr_at_fpr(self.model, self.val_loader, self.device, self.num_classes)
                fpr_results = ml_utils.evaluate_model_fpr_at_tpr(self.model, self.val_loader, self.device, self.num_classes)
                confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(self.model, self.val_loader, self.device)

            # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
            ft_training_metrics_values = {ft_metric.name: ft_metric.compute() for ft_metric in ft_training_metrics}
            ft_validation_metrics_values = {ft_metric.name: ft_metric.compute() for ft_metric in ft_validation_metrics}

            # Print metric values
            print(f"Finetuning Epoch {epoch+1}/{ft_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            ft_train_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in ft_training_metrics_values.items())
            print(f"Train Metrics | {ft_train_metrics_str}")

            ft_val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in ft_validation_metrics_values.items())
            print(f"Val Metrics | {ft_val_metrics_str}")

            wandb.log({
                "train/Loss": train_loss,
                **{
                    f"train/{name}": value.item() 
                    for name, value in ft_training_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"train/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in ft_training_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },
                "train/Learning Rate": self.optimizer.param_groups[0]['lr'],
                "train/Epoch": epoch+1,
                "validation/Loss": val_loss,
                **{
                    f"validation/{name}": value.item() 
                    for name, value in ft_validation_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"validation/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in ft_validation_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },

                **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not self.args.force_regression else {}),
                **({f"fpr_at_fixed_tpr/Average Val FPR at {tpr} TPR": np.mean(fprs) for tpr, fprs in fpr_results.items()} if not self.args.force_regression else {}),
                **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not self.args.force_regression else {}),
                **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not self.args.force_regression else {})
            })
        
        
        torch.save(self.model.state_dict(), self.model_filename)
        wandb.save(f'self.model/self.modelParameters_{self.formatted_datetime}.pth')

        # Evaluate the self.model on the test set
        ml_utils.evaluate_model_on_test_set(self.model, self.test_loader, self.
        device, self.num_gestures, self.criterion, self.args, ft_testing_metrics)

        if not self.args.force_regression:
            self.print_classification_metrics()
        self.ft_run.finish() 

    def pretrain_model(self):
        """
        Train and validation loop for the pretraining phase.
        """

        training_metrics, validation_metrics, testing_metrics = super().get_metrics()

        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            self.model.train()
            train_loss = 0.0

            # Reset training metrics at the start of each epoch
            for train_metric in training_metrics:
                train_metric.reset()
            
            if not self.args.force_regression:
                outputs_train_all = []
                ground_truth_train_all = []

            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False) as t:

                for X_batch, Y_batch in t:
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch = Y_batch.to(self.device).to(torch.float32) # ground truth

                    if self.args.force_regression:
                       Y_batch_long = Y_batch
                    else: 
                       Y_batch_long = torch.argmax(Y_batch, dim=1)

                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']
                    loss = self.criterion(output, Y_batch)
                    loss.backward()
                    self.optimizer.step()

                    if not self.args.force_regression:
                        outputs_train_all.append(output)
                        ground_truth_train_all.append(torch.argmax(Y_batch, dim=1))

                    train_loss += loss.item()

                    for train_metric in training_metrics:
                        if train_metric.name != "Macro_AUROC" and train_metric.name != "Macro_AUPRC":
                            train_metric(output,Y_batch_long)

                    if not self.args.force_regression:
                        micro_accuracy_metric = next(metric for metric in training_metrics if metric.name == "Micro_Accuracy")
                        if t.n % 10 == 0:
                            t.set_postfix({
                                "Batch Loss": loss.item(), 
                                "Batch Acc": micro_accuracy_metric.compute().item()
                            })
                
                if not self.args.force_regression:
                    outputs_train_all = torch.cat(outputs_train_all, dim=0).to(self.device)
                    ground_truth_train_all = torch.cat(ground_truth_train_all, dim=0).to(self.device)

            if not self.args.force_regression: 
                train_macro_auroc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUROC")
                train_macro_auprc_metric = next(metric for metric in training_metrics if metric.name == "Macro_AUPRC")

                train_macro_auroc_metric(outputs_train_all,   ground_truth_train_all)
                train_macro_auprc_metric(outputs_train_all, ground_truth_train_all)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            for val_metric in validation_metrics:
                val_metric.reset()

            all_val_outputs = []
            all_val_labels = []

            with torch.no_grad():
            
                for X_batch, Y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device).to(torch.float32)
                    Y_batch =Y_batch.to(self.device).to(torch.float32)
                    if self.args.force_regression:
                       Y_batch_long =Y_batch
                    else: 
                       Y_batch_long = torch.argmax(Y_batch, dim=1)

                    output = self.model(X_batch)
                    if isinstance(output, dict):
                        output = output['logits']

                    if not self.args.force_regression:
                        all_val_outputs.append(output)
                        all_val_labels.append(Y_batch_long)
                 
                    val_loss += self.criterion(output,Y_batch).item()

                    for val_metric in validation_metrics:
                        if val_metric.name != "Macro_AUROC" and val_metric.name != "Macro_AUPRC":
                            val_metric(output,Y_batch_long)

            if not self.args.force_regression:
                all_val_outputs = torch.cat(all_val_outputs, dim=0)
                all_val_labels = torch.cat(all_val_labels, dim=0)
                Y_validation_long = torch.argmax(self.Y.validation, dim=1).to(self.device)
                true_labels =Y_validation_long.cpu().detach().numpy()
                test_predictions = np.argmax(all_val_outputs.cpu().detach().numpy(), axis=1)
                conf_matrix = confusion_matrix(true_labels, test_predictions)
                print("Confusion Matrix:")
                print(conf_matrix)

                val_macro_auroc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUROC")
                val_macro_auprc_metric = next(metric for metric in validation_metrics if metric.name == "Macro_AUPRC")

                val_macro_auroc_metric(all_val_outputs,Y_validation_long)
                val_macro_auprc_metric(all_val_outputs,Y_validation_long)

            # Calculate average loss and metrics
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)

            # Compute the metrics and store them in dictionaries (to prevent multiple calls to compute)
            training_metrics_values = {metric.name: metric.compute() for metric in training_metrics}
            validation_metrics_values = {metric.name: metric.compute() for metric in validation_metrics}

            if not self.args.force_regression:
                tpr_results = ml_utils.evaluate_model_tpr_at_fpr(self.model, self.val_loader, self.device, self.num_classes)
                confidence_levels, proportions_above_confidence_threshold = ml_utils.evaluate_confidence_thresholding(self.model, self.val_loader, self.device)

            # Print metric values
            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            train_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in training_metrics_values.items())
            print(f"Train Metrics: {train_metrics_str}")

            val_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in validation_metrics_values.items())
            print(f"Val Metrics: {val_metrics_str}")

            if not self.args.force_regression: 
                for fpr, tprs in tpr_results.items():
                    print(f"Val TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
                for confidence_level, acc in confidence_levels.items():
                    print(f"Val Accuracy at {confidence_level} confidence level: {acc:.4f}")

            wandb.log({
                "train/Loss": train_loss,
                "train/Learning Rate": self.optimizer.param_groups[0]['lr'],
                "train/Epoch": epoch+1,
                "validation/Loss": val_loss,
                **{
                    f"train/{name}": value.item() 
                    for name, value in training_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"train/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in training_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },
                **{
                    f"validation/{name}": value.item() 
                    for name, value in validation_metrics_values.items() 
                    if name != 'R2Score_RawValues'
                },
                **{
                    f"validation/R2Score_RawValues_{i+1}": v.item() 
                    for name, value in validation_metrics_values.items() 
                    if name == 'R2Score_RawValues'
                    for i, v in enumerate(value)
                },
                **({f"tpr_at_fixed_fpr/Average Val TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not self.args.force_regression else {}),
                **({f"confidence_level_accuracies/Val Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not self.args.force_regression else {}),
                **({f"proportion_above_confidence_threshold/Val Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not self.args.force_regression else {}),
            })

        torch.save(self.model.state_dict(), self.model_filename)
        wandb.save(f'model/modelParameters_{self.formatted_datetime}.pth')

        # Evaluate performance on test metrics
        ml_utils.evaluate_model_on_test_set(self.model, self.test_loader, self.device, self.num_gestures, self.criterion, self.args, testing_metrics)

        if not self.args.force_regression:
            self.print_classification_metrics()
        self.pretrain_run.finish()
        

