import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass


@dataclass
class ModelArgs:
    channels: List[str]
    all_classes: List[str]
    positive_class: List[str]
    negative_class: List[str]
    reg1_order: int = 20
    topo1_n_long_living: int = 30
    classifier_base_n_out: int = 8
    fusion_method: str = "feature_level"
    topo1_weight: float = 1 / 4
    topo2_weight: float = 1 / 4
    reg1_weight: float = 1 / 4
    reg2_weight: float = 1 / 4

    def __init__(self, args):
        self.channels = args.channels
        self.all_classes = args.all_classes
        self.positive_class = args.positive_class
        self.negative_class = args.negative_class
        self.reg1_order = args.reg1_order
        self.topo1_n_long_living = args.topo1_n_long_living
        self.classifier_base_n_out = args.classifier_base_n_out
        self.fusion_method = args.fusion_method
        self.topo1_weight = args.topo1_weight
        self.topo2_weight = args.topo2_weight
        self.reg1_weight = args.reg1_weight
        self.reg2_weight = args.reg2_weight


class TopoEEG(nn.Module):
    def __init__(
        self,
        args,
        input_shapes,
        classifier_base,
    ):
        super(TopoEEG, self).__init__()

        self.fusion_method = args.fusion_method

        self.classifier_base = classifier_base
        self.classifier_base_n_out = args.classifier_base_n_out

        self.input_shapes = input_shapes
        self.feature_extractors = nn.ModuleDict()
        self._build_feature_extractors()

        if self.fusion_method == "feature_level":
            self._build_feature_level_classifier()
        elif self.fusion_method == "score_level":
            self._build_score_level_classifier()
        elif self.fusion_method == "decision_level":
            self._build_decision_models()
            self.weights = nn.Parameter(
                torch.Tensor(
                    [
                        args.topo1_weight,
                        args.topo2_weight,
                        args.reg1_weight,
                        args.reg2_weight,
                    ]
                )
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _build_CNN_1D(self, input_dim):
        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

    def _build_feature_extractors(self):
        for name, shape in self.input_shapes.items():
            self.feature_extractors[name] = self._build_CNN_1D(shape)

    def _build_feature_level_classifier(self):
        total_dim = 0
        for name, shape in self.input_shapes.items():
            total_dim += 32 * shape[0]  # 32 filters * n_features

        self.final_classifier = nn.Sequential(
            self.classifier_base(total_dim, self.classifier_base_n_out),
            nn.Linear(self.classifier_base_n_out, 1),
            nn.Sigmoid(),
        )

    def _build_score_level_classifier(self):
        self.dense_layers = nn.ModuleDict()

        for name, shape in self.input_shapes.items():
            input_dim = 32 * shape[0]  # 32 filters * n_features
            self.dense_layers[name] = self.classifier_base(
                input_dim, self.classifier_base_n_out
            )

        self.final_classifier = nn.Sequential(
            nn.Linear(self.classifier_base_n_out * len(self.input_shapes), 1),
            nn.Sigmoid(),
        )

    def _build_decision_models(self):
        self.decision_models = nn.ModuleDict()
        for name, shape in self.input_shapes.items():
            input_dim = 32 * shape[0]

            self.decision_models[name] = nn.Sequential(
                self.classifier_base(input_dim, self.classifier_base_n_out),
                nn.Linear(self.classifier_base_n_out, 1),
                nn.Sigmoid(),
            )

    def forward(self, x_dict):
        """Forward pass based on selected fusion method."""
        features = {}
        for name in self.input_shapes.keys():
            x = x_dict[name]
            if len(self.input_shapes[name]) == 1:
                x = x.unsqueeze(1)
            features[name] = self.feature_extractors[name](x)

        # 1.
        if self.fusion_method == "feature_level":
            concatenated = torch.cat(list(features.values()), dim=1)

        # 2.
        elif self.fusion_method == "score_level":
            scores = []
            for name in self.input_shapes.keys():
                scores.append(self.dense_layers[name](features[name]))
            concatenated = torch.cat(scores, dim=1)

        # 3.
        elif self.fusion_method == "decision_level":
            preds = []
            for name in self.input_shapes.keys():
                preds.append(self.decision_models[name](features[name]))
            weights = torch.softmax(self.weights, dim=0)
            weighted_avg = sum(w * p for w, p in zip(weights, preds))
            return weighted_avg

        return self.final_classifier(concatenated)
