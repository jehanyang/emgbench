import torchmetrics
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import wandb

def calculate_tpr_at_fpr(y_true, y_scores, fpr_target):
    """Calculate the TPR at a given FPR target using ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if fpr_target in fpr:
        return tpr[np.where(fpr == fpr_target)[0][0]]
    else:
        return np.interp(fpr_target, fpr, tpr)

def evaluate_model_tpr_at_fpr(model, loader, device, num_classes, fpr_targets=[0.01, 0.1, 0.5]):
    """Evaluate model to find TPR at given FPR targets for each class in multiclass classification."""
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    predictions = []
    true_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            if len(outputs) == 2:
                outputs, features = outputs
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            if len(outputs.shape) == 1:
                # outputs = outputs.unsqueeze(dim=0)
                continue
            probs = softmax(outputs).cpu().numpy()  # Get class probabilities
            predictions.append(probs)
            true_labels.append(Y_batch.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Calculate TPR at specified FPRs for each class
    tprs_at_fixed_fprs = {f"FPR {int(fpr*100)}%": [] for fpr in fpr_targets}

    # Iterate over each class
    for class_index in range(true_labels.shape[1]):
        y_true_class = true_labels[:, class_index]
        y_scores_class = predictions[:, class_index]
        
        # For each FPR target, calculate the TPR
        for fpr_target in fpr_targets:
            tpr = calculate_tpr_at_fpr(y_true_class, y_scores_class, fpr_target)
            tprs_at_fixed_fprs[f"FPR {int(fpr_target*100)}%"].append(tpr)

    return tprs_at_fixed_fprs

def calculate_fpr_at_tpr(y_true, y_scores, tpr_target):
    """Calculate the FPR at a given TPR target using ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if tpr_target in tpr:
        return fpr[np.where(tpr == tpr_target)[0][0]]
    else:
        return np.interp(tpr_target, tpr, fpr)
    
def evaluate_model_fpr_at_tpr(model, loader, device, num_classes, tpr_targets=[0.9, 0.95, 0.99]):
    """Evaluate model to find FPR at given TPR targets for each class in multiclass classification."""
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    predictions = []
    true_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            if len(outputs) == 2:
                outputs, features = outputs
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            if len(outputs.shape) == 1:
                # outputs = outputs.unsqueeze(dim=0)
                continue
            probs = softmax(outputs).cpu().numpy()  # Get class probabilities
            predictions.append(probs)
            true_labels.append(Y_batch.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Calculate FPR at specified TPRs for each class
    fprs_at_fixed_tprs = {f"TPR {int(tpr*100)}%": [] for tpr in tpr_targets}

    # Iterate over each class
    for class_index in range(true_labels.shape[1]):
        y_true_class = true_labels[:, class_index]
        y_scores_class = predictions[:, class_index]
        
        # For each TPR target, calculate the FPR
        for tpr_target in tpr_targets:
            fpr = calculate_fpr_at_tpr(y_true_class, y_scores_class, tpr_target)
            fprs_at_fixed_tprs[f"TPR {int(tpr_target*100)}%"].append(fpr)

    return fprs_at_fixed_tprs

def evaluate_confidence_thresholding(model, loader, device, thresholds=[0.5, 0.9, 0.95, 0.99]):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    correct_counts = {thresh: 0 for thresh in thresholds}
    total_counts = {thresh: 0 for thresh in thresholds}
    total_above_threshold_counts = {thresh: 0 for thresh in thresholds}

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            if len(outputs) == 2:
                outputs, features = outputs
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            if len(outputs.shape) == 1:
                # outputs = outputs.unsqueeze(dim=0)
                continue
            probs = softmax(outputs)
            max_probs, preds = torch.max(probs, dim=1)

            for thresh in thresholds:
                above_thresh = max_probs > thresh
                total_counts[thresh] += above_thresh.sum().item()
                correct_counts[thresh] += (preds[above_thresh] == torch.argmax(Y_batch[above_thresh], axis=1)).sum().item()
                total_above_threshold_counts[thresh] += len(max_probs[max_probs > thresh])

    confidence_accuracy = {thresh: (correct_counts[thresh] / total_counts[thresh]) if total_counts[thresh] > 0 else 0 for thresh in thresholds}
    proportion_above_threshold = {thresh: (total_above_threshold_counts[thresh] / len(loader.dataset)) for thresh in thresholds}

    return confidence_accuracy, proportion_above_threshold

def evaluate_model_on_test_set(model, test_loader, device, numGestures, criterion, args, testing_metrics):

    # Assuming model, criterion, device, and test_loader are defined
    model.eval()
    test_loss = 0.0

    # Reset test metrics
    for test_metric in testing_metrics:
        test_metric.reset()

    pred = []
    true = []
    outputs_all = []
    Y_test = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)
            Y_test.append(Y_batch)

            if args.force_regression:
                Y_batch_long = Y_batch
            else: 
                Y_batch_long = torch.argmax(Y_batch, dim=1) 

            output = model(X_batch)
            if len(output) == 2:
                output, features = output
            if isinstance(output, dict):
                output = output['logits']
            if len(output.shape) == 1:
                # output = output.unsqueeze(dim=0)
                continue
            pred.extend(torch.argmax(output, dim=1).cpu().detach().numpy())
            true.extend(Y_batch_long.cpu().detach().numpy())

            if not args.force_regression:
                outputs_all.append(output)

            test_loss += criterion(output, Y_batch).item()

        outputs_all = torch.cat(outputs_all, dim=0).to(device)
        Y_test = torch.cat(Y_test, dim=0).to(device)
        Y_test = torch.argmax(Y_test, dim=1)


        for test_metric in testing_metrics:
            if test_metric.name != "Macro_AUROC" and test_metric.name != "Macro_AUPRC":
                test_metric(outputs_all, Y_test)

            

    if not args.force_regression: 
        true_torch = torch.tensor(true).to(device)

        test_macro_auroc = next(metric for metric in testing_metrics if metric.name == "Macro_AUROC")
        test_macro_auprc = next(metric for metric in testing_metrics if metric.name == "Macro_AUPRC")

        test_macro_auroc(outputs_all, true_torch)
        test_macro_auprc(outputs_all, true_torch)

    # Calculate average loss and metrics
    test_loss /= len(test_loader)

    testing_metrics_values = {metric.name: metric.compute() for metric in testing_metrics}

    if not args.force_regression: 
        tpr_results = evaluate_model_tpr_at_fpr(model, test_loader, device, numGestures)
        fpr_results = evaluate_model_fpr_at_tpr(model, test_loader, device, numGestures)
        confidence_levels, proportions_above_confidence_threshold = evaluate_confidence_thresholding(model, test_loader, device)


    testing_metrics_str = " | ".join(f"{name}: {value.item():.4f}" if name != 'R2Score_RawValues' else f"{name}: ({', '.join(f'{v.item():.4f}' for v in value)})" for name, value in testing_metrics_values.items())

    print(f"Test Metrics: {testing_metrics_str}")

    wandb.log({
        "test/Loss": test_loss,
        **{
            f"test/{name}": value.item() 
            for name, value in testing_metrics_values.items() 
            if name != 'R2Score_RawValues'
        },
        **{
            f"test/R2Score_RawValues_{i+1}": v.item() 
            for name, value in testing_metrics_values.items() 
            if name == 'R2Score_RawValues'
            for i, v in enumerate(value)
        },
        # **{f"tpr_at_fixed_fpr/Test TPR at {fpr} FPR - Gesture {idx}": tpr for fpr, tprs in tpr_results.items() for idx, tpr in enumerate(tprs)},
        **({f"tpr_at_fixed_fpr/Average Test TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()} if not args.force_regression else {}),
        **({f"fpr_at_fixed_tpr/Average Test FPR at {tpr} TPR": np.mean(fprs) for tpr, fprs in fpr_results.items()} if not args.force_regression else {}),
        **({f"confidence_level_accuracies/Test Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()} if not args.force_regression else {}),
        **({f"proportion_above_confidence_threshold/Test Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()} if not args.force_regression else {})
    })

