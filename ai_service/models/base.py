"""
Base class interface for all trainable models.
Every model registered in the registry must implement this interface.
"""

from ai_service.dirs import MODEL_DIR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import torch
import timm
from timm.data import resolve_data_config, create_transform
from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler_v2
from timm.optim import create_optimizer_v2
from sklearn.metrics import accuracy_score, average_precision_score
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np

class TrainableModel:
    """Base class for image detection/classification models."""

    def __init__(self, num_classes: int, model_name: str | None = None, device: str | None = None,
                 weights_path: str | None = None, pretrained: bool = True, drop_rate: float = 0.0):
        """
        Initializes the model. If weights_path is provided, it loads the state_dict for inference or export.
        Uses the 'timm_name' class attribute as a fallback when model_name is not provided (registry usage).
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name or getattr(self, 'timm_name', None)
        if not self.model_name:
            raise ValueError("Either 'model_name' argument or 'timm_name' class attribute must be set.")
        self.num_classes = num_classes

        # 1. Build Model Architecture
        # If loading existing weights, we don't need timm to download pretrained weights
        load_pretrained = pretrained and (weights_path is None)
        self.model = timm.create_model(
            self.model_name,
            pretrained=load_pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        ).to(self.device)

        # 2. Load State Dict if requested (Inference/Export Mode)
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # Set to evaluation mode immediately

        # 3. Resolve Data Config dynamically
        self.data_config = resolve_data_config(self.model.pretrained_cfg)
        self.val_transform: transforms.Compose = create_transform(**self.data_config, is_training=False) #type:ignore

        # Training state variables
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.ema = None
        self.mixup_fn = None
        self.criterion_train: nn.Module = nn.CrossEntropyLoss()
        self.criterion_val = nn.CrossEntropyLoss() # Validation never uses soft targets

    def setup_training(self, opt_name='adamw', lr=1e-3, weight_decay=0.05,
                       sched_name='cosine', epochs=10, warmup_epochs=5,
                       use_ema=True, mixup_alpha=0.0, cutmix_alpha=0.0):
        self.optimizer = create_optimizer_v2(self.model, opt=opt_name, lr=lr, weight_decay=weight_decay)
        self.scheduler, _ = create_scheduler_v2(self.optimizer, sched=sched_name, num_epochs=epochs, warmup_epochs=warmup_epochs)

        if use_ema:
            self.ema = ModelEmaV2(self.model, decay=0.9998)

        if mixup_alpha > 0.0 or cutmix_alpha > 0.0:
            self.mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_classes=self.num_classes)
            self.criterion_train = SoftTargetCrossEntropy()

    def train_one_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss = 0.0

        assert self.optimizer is not None, "Optimizer has not been initialized. Call setup_training() first."

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.mixup_fn is not None:
                inputs, targets = self.mixup_fn(inputs, targets)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion_train(outputs, targets)
            loss.backward()
            self.optimizer.step()

            if self.ema:
                self.ema.update(self.model)
            total_loss += loss.item()

        if self.scheduler:
            self.scheduler.step(epoch)
        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = eval_model(inputs)

            loss = self.criterion_val(outputs, targets)
            total_loss += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Calculate Metrics
        val_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_targets, all_preds)

        # Calculate mAP (Macro Average Precision) using one-hot encoded targets
        targets_one_hot = np.eye(self.num_classes)[all_targets]
        map_score = average_precision_score(targets_one_hot, all_probs, average="macro")

        return {"val_loss": val_loss, "accuracy": acc, "mAP": map_score}

    @torch.no_grad()
    def inference(self, image_path: str, class_mapping: dict, threshold: float = 0.5):
        """
        Fast inference returning only classes that exceed the confidence threshold.
        """
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.val_transform(img).unsqueeze(0).to(self.device)

        outputs = self.model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        results = {}
        # Iterate over all predictions and filter by threshold
        for idx, prob_tensor in enumerate(probabilities):
            prob: float = prob_tensor.item()
            if prob >= threshold:
                # Find the class name corresponding to this index
                class_name: str = next(name for name, i in class_mapping.items() if i == idx)
                results[class_name] = round(prob, 4)

        # Sort results by confidence descending
        return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    def save_model(self, model_id: str, class_mapping: dict):
        """Saves the PyTorch weights and the class mapping JSON."""
        save_model = self.ema.module if self.ema is not None else self.model
        weight_path = os.path.join(MODEL_DIR, f"{model_id}.pt")
        map_path = os.path.join(MODEL_DIR, f"{model_id}_map.json")

        torch.save(save_model.state_dict(), weight_path)
        with open(map_path, 'w') as f:
            json.dump(class_mapping, f)

        return weight_path, map_path

    def export(self, export_format: str, export_path: str):
        """Exports the model to the requested format."""
        input_size = self.data_config['input_size']
        dummy_input = torch.randn(1, *input_size, device=self.device)

        if export_format == "onnx":
            torch.onnx.export(
                self.model, (dummy_input,), export_path, export_params=True,
                opset_version=13, do_constant_folding=True,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

        elif export_format == "torchscript":
            traced_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(traced_model, export_path)

        elif export_format == "tflite":
            # TFLite requires a bridge. The standard modern approach is exporting to ONNX,
            # then using 'onnx2tf' library (CLI) to generate a SavedModel, then converting to TFLite.
            # In a real microservice, you would call `subprocess.run()` here.
            onnx_tmp_path = export_path.replace(".tflite", ".onnx")
            self.export("onnx", onnx_tmp_path)
            # pseudo-code for system call:
            # os.system(f"onnx2tf -i {onnx_tmp_path} -o {export_path}_saved_model")
            # os.system(f"tflite_convert --saved_model_dir={export_path}_saved_model --output_file={export_path}")
            raise NotImplementedError("TFLite export requires ONNX->TF system subprocess compilation.")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        return export_path
