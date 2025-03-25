# Copyright (c) EEEM071, University of Surrey

import wandb
import uuid
import time
import os
from dotenv import load_dotenv

class WandBLogger:
    def __init__(self, args=None):
        self.enabled = args is not None and hasattr(args, 'use_wandb') and args.use_wandb
        self.args = args
        
        if self.enabled and not args.evaluate:
            load_dotenv()
            augmentations = []
            # by default, horizontal flips and translations are always applied
            if args.random_erase: augmentations.append("erase")
            if args.color_jitter: augmentations.append("jitter")
            if args.color_aug: augmentations.append("color")
            aug_name = "+".join(augmentations) if augmentations else "base"
            
            student_id = os.getenv('STUDENT_ID')
            student_name = os.getenv('STUDENT_NAME')
            
            wandb.init(
                project="VeRi",
                name=f"{args.arch}_{aug_name}_{args.max_epoch}",
                config=vars(args),
            )
            wandb.run.summary["student_id"] = student_id
            wandb.run.summary["student_name"] = student_name
            wandb.run.summary["uuid"] = str(uuid.uuid4())
            wandb.run.summary["experiment_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    def watch_model(self, model):
        """Watch model parameters and gradients"""
        if self.enabled and not self.args.evaluate:
            wandb.watch(model, log="all", log_freq=100)
    
    def log_train_metrics(self, metrics, epoch, batch_idx, trainloader_len):
        """Log training metrics with step based on epoch and batch index"""
        if self.enabled:
            step = epoch * trainloader_len + batch_idx
            wandb.log(metrics, step=step)
    
    def log_test_metrics(self, metrics):
        """Log testing/evaluation metrics"""
        if self.enabled and not self.args.evaluate:  # Only log during training runs
            wandb.log(metrics)
    
    def format_train_metrics(self, xent_loss, htri_loss, accuracy, learning_rate, lambda_xent, lambda_htri):
        """Format training metrics for logging"""
        return {
            "train/xent_loss": xent_loss,
            "train/htri_loss": htri_loss,
            "train/total_loss": xent_loss * lambda_xent + htri_loss * lambda_htri,
            "train/accuracy": accuracy,
            "train/learning_rate": learning_rate,
        }
    
    def format_test_metrics(self, mAP, cmc):
        """Format test metrics for logging"""
        return {
            "test/mAP": mAP * 100,
            "test/rank1": cmc[0] * 100,
            "test/rank5": cmc[4] * 100,
            "test/rank10": cmc[9] * 100,
            "test/rank20": cmc[19] * 100,
        }
    
    def finish(self):
        """End the wandb run"""
        if self.enabled:
            wandb.finish()
