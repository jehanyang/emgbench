"""
Model_Trainer.py
- Superclass for all model trainers. 
"""
from torch.utils.data import Dataset
import gc
from torchvision import transforms
from PIL import Image
import wandb
import os 
import torch.nn as nn
import torchmetrics
import numpy as np
import torch 
import multiprocessing
from torch.utils.data import DataLoader
import random
import copy
from torch.utils.data import Sampler
import math



class Model_Trainer():
    def __init__(self, X_data, Y_data, label_data, env):
        self.X = X_data
        self.Y = Y_data
        self.label = label_data

        self.env = env

        self.args = env.args
        self.utils = env.utils
        self.num_gestures = env.num_gestures
        self.project_name = env.project_name
        self.formatted_datetime = env.formatted_datetime
        self.exercises = env.exercises

        # Set seeds for reproducibility
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if hasattr(self.X, 'leaveOutIndices'):
            self.leaveOutIndices = self.X.leaveOutIndices

        self.pretrain_path = None
        self.criterion = None
        self.num_epochs = self.args.epochs
        self.run = None
        self.device = None
        self.testrun_foldername = None
        self.model_filename = None
        self.gesture_labels = None
        self.num_classes = None

        # Defined in either CNN_Trainer, Classic_Trainer
        if not self.args.turn_on_unlabeled_domain_adaptation:
            self.model_name = self.args.model
            self.model = None 
            self.batch_size = self.args.batch_size

            self.train_loader = None
            self.val_loader = None
            self.test_loader = None

            self.train_dataset = None # for CNN loop

        else:
            self.train_labeled_loader = None 
            self.train_unlabeled_loader = None
            self.train_finetuning_loader = None
            self.train_finetuning_unlabeled_loader = None
            self.validation_loader = None
            self.test_loader = None
            self.iters_for_loader = None

        # Temporary helpers
        self.resize_transform = None 
        self.scheduler = None

        self.training_metrics = None
        self.validation_metrics = None
        self.testing_metrics = None

        self.wandb_runname = None

        self.device = torch.device("cuda:" + str(self.args.gpu) if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.set_num_classes()
        
        

    class CustomDataset(Dataset):
        def __init__(self, X, Y, transform=None):
            self.X = X
            self.Y = Y
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            x = self.X[index]
            y = self.Y[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        
    def ceildiv(a, b):
        return -(a // -b)

    def set_pretrain_path(self):
        if self.args.model == "vit_tiny_patch2_32":
            self.pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth"
        elif self.args.model == "resnet50":
            self.pretrain_path = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        else:
            self.pretrain_path = f"https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/{self.model_name}_mlp_im_1k_224.pth"

    def get_metrics(self, testing=True):
        """
        Constructs training and validation metric arrays based on whether it is a regression or classification task. Also returns testing metrics based on testing flag.

        All changes to metrics should be done here. 
        """
        def get_regression_metrics():
            regression_metrics = [
                torchmetrics.MeanSquaredError().to(self.device), 
                torchmetrics.MeanSquaredError(squared=False).to(self.device), 
                torchmetrics.MeanAbsoluteError().to(self.device),
                torchmetrics.R2Score(num_outputs=6, multioutput="uniform_average").to(self.device), 
                torchmetrics.R2Score(num_outputs=6, multioutput="raw_values").to(self.device)
            ]
            for metric, name in zip(regression_metrics, ["MeanSquaredError", "RootMeanSquaredError","MeanAbsoluteError", "R2Score_UniformAverage", "R2Score_RawValues"]):
                metric.name = name

            return regression_metrics

        def get_classification_metrics():
            classification_metrics = [

                torchmetrics.Accuracy(task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device),
                torchmetrics.Precision(task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device),
                torchmetrics.Recall(task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device),
                torchmetrics.F1Score(task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device),
                torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device),
                torchmetrics.Accuracy(task="multiclass", num_classes=self.num_gestures, average="micro").to(self.device),
                torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=self.num_gestures, average="micro").to(self.device),
                torchmetrics.AUROC(task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device),
                torchmetrics.AveragePrecision(task="multiclass", num_classes=self.num_gestures, average="macro").to(self.device)
            ]
            for metric, name in zip(classification_metrics, ["Macro_Acc", "Macro_Precision", "Macro_Recall", "Macro_F1Score", "Macro_Top5Accuracy", "Micro_Accuracy", "Micro_Top5Accuracy", "Macro_AUROC","Macro_AUPRC"]):
                metric.name = name
            
            return classification_metrics
        

        def get_transition_classifier_metrics():
            classification_metrics = [

                torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device),
                torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device),
                torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device),
                torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device),
                torchmetrics.Accuracy(top_k=1, task="multiclass", num_classes=self.num_classes, average="macro").to(self.device),
                torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, average="micro").to(self.device),
                torchmetrics.Accuracy(top_k=1, task="multiclass", num_classes=self.num_classes, average="micro").to(self.device),
                torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device),
                torchmetrics.AveragePrecision(task="multiclass", num_classes=self.num_classes, average="macro").to(self.device)
            ]
            for metric, name in zip(classification_metrics, ["Macro_Acc", "Macro_Precision", "Macro_Recall", "Macro_F1Score", "Macro_Top5Accuracy", "Micro_Accuracy", "Micro_Top5Accuracy", "Macro_AUROC","Macro_AUPRC"]):
                metric.name = name
            
            return classification_metrics
        

         
        if self.args.force_regression:
            training_metrics = get_regression_metrics()
            validation_metrics = get_regression_metrics()
            testing_metrics = get_regression_metrics()

        elif self.args.transition_classifier:
            training_metrics = get_transition_classifier_metrics()
            validation_metrics = get_transition_classifier_metrics()
            testing_metrics = get_transition_classifier_metrics()

        else: 
            training_metrics = get_classification_metrics()
            validation_metrics = get_classification_metrics()
            testing_metrics = get_classification_metrics()

        if not testing:
            return training_metrics, validation_metrics
        return training_metrics, validation_metrics, testing_metrics
    
    def setup_model(self):
        raise NotImplementedError("Subclasses (CNN_Trainer, CNN_LSTM_Trainer, Transformer_Trainer) must implement setup_model()")

    class ToNumpy:
        """Custom transformation to convert PIL Images or Tensors to NumPy arrays."""
        def __call__(self, pic):
            if isinstance(pic, Image.Image):
                return np.array(pic)
            elif isinstance(pic, torch.Tensor):
                # Make sure the tensor is in CPU and convert it
                return np.float32(pic.cpu().detach().numpy())
            else:
                raise TypeError("Unsupported image type")

    def set_resize_transform(self):
        """
        Helper function for setup_model(). Sets the resize_transform attribute based on the model type.

        Returns:
            _type_: _description_
        """

        class ToVector:
        
            def __call__(self, img):
                # Convert image to a tensor and flatten it
                return img.flatten()

        if self.args.model == 'vit_tiny_patch2_32':
            resize_transform = transforms.Compose([transforms.Resize((32,32)), self.ToNumpy()])
        else:
            resize_transform = transforms.Compose([transforms.Resize((224,224)), self.ToNumpy()])
            if self.args.model == "MLP":
                resize_transform = transforms.Compose([resize_transform, ToVector()])

        self.resize_transform = resize_transform

    def create_datasets(self):

        if self.args.turn_on_unlabeled_domain_adaptation:
            raise NotImplementedError("This method should be overwritten in Unlabelled_Domain_Adaptation_Trainer")

        else:

            self.train_dataset = self.CustomDataset(
                self.X.train, 
                self.Y.train, 
                transform=self.resize_transform
            )

            val_dataset = self.CustomDataset(
                self.X.validation, 
                self.Y.validation, 
                transform=self.resize_transform
            )

            test_dataset = self.CustomDataset(
                self.X.test, 
                self.Y.test, 
                transform=self.resize_transform
            )

            return self.train_dataset, val_dataset, test_dataset

    def set_loaders(self):

        if self.args.turn_on_unlabeled_domain_adaptation:
            raise NotImplementedError("This method should be overwritten in Unlabelled_Domain_Adaptation_Trainer")
        
        elif self.args.domain_generalization in {"IRM", "CORAL"}:

            train_dataset, val_dataset, test_dataset = self.create_datasets()

            # Randomly sample within batches 
            self.sampler = RandomDomainSampler(
                batch_size=self.args.batch_size,
                cumulative_sizes=self.X.cumulative_sizes,
                num_subjects=self.utils.num_subjects
            )

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=multiprocessing.cpu_count()//8,
                worker_init_fn=self.utils.seed_worker,
                pin_memory=True,
                sampler=self.sampler,
                # drop_last=True 
                drop_last=False
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=multiprocessing.cpu_count()//8,
                worker_init_fn=self.utils.seed_worker,
                pin_memory=True,
                shuffle=False,
                # drop_last=True
                drop_last=False
            )

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.args.batch_size,
                num_workers=multiprocessing.cpu_count()//8,
                worker_init_fn=self.utils.seed_worker,
                pin_memory=True,
                shuffle=False,
                # drop_last=True
                drop_last=False
            )



        else:

            train_dataset, val_dataset, test_dataset = self.create_datasets()

            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=multiprocessing.cpu_count() // 8, 
                worker_init_fn=self.utils.seed_worker, 
                pin_memory=True,
                drop_last=self.args.force_regression
            )
            
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                num_workers=multiprocessing.cpu_count() // 8, 
                worker_init_fn=self.utils.seed_worker, 
                pin_memory=True, 
                drop_last=self.args.force_regression
            )

            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                num_workers=multiprocessing.cpu_count() // 8, 
                worker_init_fn=self.utils.seed_worker, 
                pin_memory=True, 
                drop_last=self.args.force_regression
            )

    def set_criterion(self):

        # TODO: Add in CORAL/IRM

        if self.args.force_regression:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def clear_memory(self):
        assert not self.args.turn_on_unlabeled_domain_adaptation, "clear_memory() only for non UDA models"
        # Training loop
        gc.collect()
        torch.cuda.empty_cache()

    
    def set_wandb_runname(self):
        wandb_runname = 'CNN_seed-'+str(self.args.seed)

        if self.args.transition_classifier:
            wandb_runname += '_transition_classifier'
        if self.args.turn_on_rms:
            wandb_runname += '_rms-'+str(self.args.rms_input_windowsize)
        if self.args.leftout_subject != 0:
            if (self.args.force_regression and self.args.dataset == "ninapro-db3") and self.args.leftout_subject == 10:
                # Subject 10 in DB3 is missing a lot of data. We delete it internally and subject 11 gets shifted to become subject 10. Naming it its the "external" subject number for consistency. 
                wandb_runname += '_LOSO-11'

            wandb_runname += '_LOSO-'+str(self.args.leftout_subject)
        wandb_runname += '_' + self.model_name
        if (self.exercises and not self.args.partial_dataset_ninapro):
            wandb_runname += '_exer-' + ''.join(character for character in str(self.args.exercises) if character.isalnum())
        if self.args.dataset == "mcs":
            if self.args.full_dataset_mcs:
                wandb_runname += '_full'
            else:
                wandb_runname += '_partial'
        if self.args.dataset == "ninapro-db2" or self.args.dataset == "ninapro-db5":
            if self.args.partial_dataset_ninapro:
                wandb_runname += '_partial'
        if self.args.turn_on_spectrogram:
            wandb_runname += '_spect'
        if self.args.turn_on_cwt:
            wandb_runname += '_cwt'
        if self.args.turn_on_hht:
            wandb_runname += '_hht'
        if self.args.turn_on_phase_spectrogram:
            wandb_runname += '_phase-spect'
        if self.args.reduce_training_data_size:
            wandb_runname += '_reduced-training-data-size-' + str(self.args.reduced_training_data_size)
        if self.args.turn_off_scaler_normalization:
            wandb_runname += '_no-scal-norm'
        if self.args.target_normalize > 0:
            wandb_runname += '_targ-norm-' + str(self.args.target_normalize)
        if self.args.transfer_learning:
            wandb_runname += '_tran-learn'
            wandb_runname += '-prop-' + str(self.args.proportion_transfer_learning_from_leftout_subject)
        if self.args.train_test_split_for_time_series:   
            wandb_runname += '_cv-for-ts'
        if self.args.reduce_data_for_transfer_learning != 1:
            wandb_runname += '_red-data-for-tran-learn-' + str(self.args.reduce_data_for_transfer_learning)
        if self.args.leave_one_session_out:
            wandb_runname += '_leave-one-sess-out'
        if self.args.leave_one_subject_out:
            wandb_runname += '_loso'
        if self.args.one_subject_for_training_set_for_session_test:
            wandb_runname += '_one-subj-for-training-set'
        if self.args.pretrain_and_finetune:
            wandb_runname += '_pretrain-finetune'
        if self.args.turn_on_unlabeled_domain_adaptation:
            wandb_runname += '_unlabeled-adapt'
            wandb_runname += '-algo-' + self.args.unlabeled_algorithm
            wandb_runname += '-prop-unlabel-leftout' + str(self.args.proportion_unlabeled_data_from_leftout_subject)
        if self.args.proportion_data_from_training_subjects<1.0:
            wandb_runname += '_train-subj-prop-' + str(self.args.proportion_data_from_training_subjects)
        if self.args.proportion_unlabeled_data_from_training_subjects>0:
            wandb_runname += '_unlabel-subj-prop-' + str(self.args.proportion_unlabeled_data_from_training_subjects)

        if (self.args.target_normalize > 0) and (self.args.target_normalize_subject != self.args.leftout_subject):
            wandb_runname += '_targ-norm-subj-' + str(self.args.target_normalize_subject)

        if self.args.include_transitions: 
            wandb_runname += '_include-transitions'

        self.wandb_runname = wandb_runname


    def set_project_name(self):
        if self.args.leave_one_subject_out:
            self.project_name += '_LOSO'
        elif self.args.leave_one_session_out:
           self.project_name += '_leave-one-session-out'

        self.project_name += self.args.project_name_suffix


    def start_pretrain_run(self):

        self.clear_memory()
        self.set_wandb_runname()
        self.set_project_name()

        self.pretrain_run = wandb.init(name=self.wandb_runname, project=self.project_name)
        wandb.config.lr = self.args.learning_rate
    
    def set_model_to_device(self):
        
        if not self.args.turn_on_unlabeled_domain_adaptation and self.args.model not in ['MLP', 'SVC', 'RF']:
            self.model.to(self.device)
            wandb.watch(self.model)

    def set_testrun_foldername(self):
        self.testrun_foldername = f'test/{self.project_name}/{self.wandb_runname}/{self.formatted_datetime}/'
        # Make folder if it doesn't exist
        if not os.path.exists(self.testrun_foldername):
            os.makedirs(self.testrun_foldername)
        
        self.model_filename = f'{self.testrun_foldername}model_{self.formatted_datetime}.pth'

    def set_gesture_labels(self):

        if self.args.transition_classifier:
            self.gesture_labels = self.utils.transition_labels
        else: 
            if (self.exercises):
                if not self.args.partial_dataset_ninapro:
                    self.gesture_labels = self.utils.gesture_labels['Rest']
                    for exercise_set in self.args.exercises:
                        self.gesture_labels = self.gesture_labels + self.utils.gesture_labels[exercise_set]
                else:
                    self.gesture_labels = self.utils.partial_gesture_labels
            else:
                self.gesture_labels = self.utils.gesture_labels

    def plot_test_images(self):
        self.utils.plot_average_images(self.X.test, np.argmax(self.label.test.cpu().detach().numpy(), axis=1), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'test')
        self.utils.plot_first_fifteen_images(self.X.test, np.argmax(self.label.test.cpu().detach().numpy(), axis=1), self.gesture_labels, self.testrun_foldername, self.args, self.formatted_datetime, 'test')
        

    def plot_validation_images(self):

        self.utils.plot_average_images(
            self.X.validation, 
            np.argmax(self.label.validation.cpu().detach().numpy(), axis=1), 
            self.gesture_labels, 
            self.testrun_foldername, 
            self.args, 
            self.formatted_datetime, 
            'gesture validation'
        )
     
    def plot_train_images(self):

        self.utils.plot_average_images(
            self.X.train, 
            np.argmax(self.label.train.cpu().detach().numpy(), axis=1), 
            self.gesture_labels, 
            self.testrun_foldername, 
            self.args, 
            self.formatted_datetime, 
            'gesture train'
        )

    def plot_finetuning_images(self):
        self.utils.plot_average_images(
            self.X.train_finetuning, 
            np.argmax(self.label.train_finetuning.cpu().detach().numpy(), axis=1), 
            self.gesture_labels, 
            self.testrun_foldername, 
            self.args, 
            self.formatted_datetime, 
            'gesture train_finetuning'
        )

        self.utils.plot_first_fifteen_images(
            self.X.train_finetuning, 
            np.argmax(self.label.train_finetuning.cpu().detach().numpy(), axis=1), 
            self.gesture_labels, 
            self.testrun_foldername, 
            self.args, 
            self.formatted_datetime, 
            'gesture train_finetuning'
        )

    def plot_images(self):
    
        self.plot_test_images()
        self.plot_validation_images()
        self.plot_train_images()
        if self.args.pretrain_and_finetune:
            self.plot_finetuning_images()

    def set_num_classes(self):
        if self.args.force_regression:
            num_classes = self.Y.train.shape[1]
            # assert num_classes == 6
        elif self.args.transition_classifier:
            num_classes = 2
        else: 
            num_classes = self.num_gestures

        self.num_classes = num_classes



    def shared_setup(self):
        # set up function calls shared for all models 

        self.set_wandb_runname()
        self.initialize_wandb()
        self.set_device()
        self.set_testrun_folder()
        self.set_gesture_labels()
        self.plot_images()
        self.set_num_classes()
        self.set_criterion()

    def model_loop(self):
        raise NotImplementedError("Subclasses (Unlabeled_Domain_Adaptation, MLP, or SVC_RF trainers) must implement model_loop()")
    


'''

RandomDomainSampler from: https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_generalization/image_classification/irm.py
'''

class RandomDomainSampler(Sampler):
    """
    Source: DomainBed 
    Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N\times K`.

    Args:
        data_source: dataset that contains data from multiple domains (each subject's data is concated to itself)
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times K` here)
    """

    def __init__(self, batch_size: int, cumulative_sizes, num_subjects):

        super(Sampler, self).__init__()
        self.n_domains_in_dataset = num_subjects - 1
        self.n_domains_per_batch = num_subjects - 1 

        self.sample_idxes_per_domain = [] 

        self.batch_size_per_domain = []
        total_windows = cumulative_sizes[-1]

        start = 0
        # Sort indices for each domain (subject) 
        for end in cumulative_sizes:
            subject_proportion = math.ceil((end-start) / total_windows * batch_size)
            self.batch_size_per_domain.append(subject_proportion)

            idxes = [idx for idx in range(start, end)] 
            self.sample_idxes_per_domain.append(idxes) 
            start = end

      
        total_batch_size = sum(self.batch_size_per_domain)

        assert total_batch_size >= batch_size, f"Total batch size {total_batch_size} is less than batch size {batch_size}. Rounding issue. "
        while total_batch_size != batch_size: 
            max_idx = self.batch_size_per_domain.index(max(self.batch_size_per_domain))
            self.batch_size_per_domain[max_idx] -= 1
            total_batch_size = sum(self.batch_size_per_domain)

        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)] # a list of all the domains we can pick from
        final_idxes = []
        stop_flag = False

        while not stop_flag:
            # shuffle select for the domain 
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch) 

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                
                # if not enough, sample with replacement
                if len(sample_idxes) < self.batch_size_per_domain[domain]:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain[domain], replace=True)
                # otherwise, sample without replacement
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain[domain])
                final_idxes.extend(selected_idxes)

                # Remove the indices we chose
                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain[domain]:
                    stop_flag = True
        
        proportion_in_order = []
        for domain in selected_domains: 
            proportion_in_order.append(self.batch_size_per_domain[domain])

        self.curr_prop_per_domain = proportion_in_order

        return iter(final_idxes)

    def __len__(self):
        return self.length
    
    def get_prop_per_domain(self):
        return self.curr_prop_per_domain



        

        

            
            
