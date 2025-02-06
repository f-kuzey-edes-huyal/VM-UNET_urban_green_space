import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    Train model for one epoch
    '''
    # Switch to train mode
    model.train() 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        # Convert targets to binary format: 1 where max probability, 0 otherwise
        targets = (targets == targets.max(dim=1, keepdim=True).values).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    # Scheduler step (metric-based schedulers like ReduceLROnPlateau need validation loss)
    scheduler.step(np.mean(loss_list))  # For metric-based schedulers, pass the average loss here
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion, 
                  epoch, 
                  logger,
                  config,
                  scheduler):  
    '''
    Validate model for one epoch
    '''
    # Switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # Convert masks to binary format
            msk = (msk == msk.max(dim=1, keepdim=True).values).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.cpu().detach().numpy())

            if isinstance(out, tuple):
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

    # Compute the mean validation loss and call the scheduler step
    val_loss = np.mean(loss_list)
    scheduler.step(val_loss)

    log_info = f'val epoch: {epoch}, loss: {val_loss:.4f}'
    print(log_info)
    logger.info(log_info)

    return val_loss


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    '''
    Test model for one epoch
    '''
    # Switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # Convert masks to binary format
            msk = (msk == msk.max(dim=1, keepdim=True).values).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.cpu().detach().numpy())

            if isinstance(out, tuple):
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        # Compute confusion matrix and metrics
        preds = np.array(preds).reshape(-1, 5)  # Shape: (H*W, num_classes)
        gts = np.array(gts).reshape(-1, 5)      # Shape: (H*W, num_classes)

        y_pre = np.argmax(preds, axis=1)  # Predicted class index (0-4)
        y_true = np.argmax(gts, axis=1)   # True class index (0-4)

        assert np.all(np.isin(y_pre, np.arange(5))), "Predictions contain invalid class labels!"
        assert np.all(np.isin(y_true, np.arange(5))), "Ground truth contains invalid class labels!"

        # Compute confusion matrix
        confusion = confusion_matrix(y_true, y_pre, labels=np.arange(5))
        print("Confusion Matrix:\n", confusion)

        # Compute per-class metrics
        ious, f1_scores, sensitivities, specificities, accuracies = [], [], [], [], []

        for i in range(5):  
            TP = confusion[i, i]
            FP = confusion[:, i].sum() - TP
            FN = confusion[i, :].sum() - TP
            TN = confusion.sum() - (TP + FP + FN)

            # Metric calculations
            acc = (TP + TN) / confusion.sum() if confusion.sum() != 0 else 0
            sens = TP / (TP + FN) if (TP + FN) != 0 else 0
            spec = TN / (TN + FP) if (TN + FP) != 0 else 0
            f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

            ious.append(iou)
            f1_scores.append(f1)
            sensitivities.append(sens)
            specificities.append(spec)
            accuracies.append(acc)

        # Compute mean metrics
        mean_iou = np.mean(ious)
        mean_f1 = np.mean(f1_scores)
        mean_sensitivity = np.mean(sensitivities)
        mean_specificity = np.mean(specificities)
        mean_accuracy = np.mean(accuracies)

        # Log results
        log_info = (
        f"Test Results -> Mean IoU: {mean_iou:.4f}, Mean F1 Score: {mean_f1:.4f}, "
        f"Mean Accuracy: {mean_accuracy:.4f}, "
        f"Mean Specificity: {mean_specificity:.4f}, Mean Sensitivity: {mean_sensitivity:.4f}"
        )
        print(log_info)
        logger.info(log_info)

        # Return mean of the loss list
        return np.mean(loss_list)
