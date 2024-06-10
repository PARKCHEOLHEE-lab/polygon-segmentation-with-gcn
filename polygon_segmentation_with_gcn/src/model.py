import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import shapely
import torch
import torch.nn as nn
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ray

from PIL import Image
from shapely import geometry, ops
from typing import List, Tuple
from tqdm import tqdm
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, GraphConv, GATConv
from torch_geometric.utils import negative_sampling
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, AUROC, Recall

from polygon_segmentation_with_gcn.src.commonutils import runtime_calculator, add_debugvisualizer
from polygon_segmentation_with_gcn.src.config import Configuration
from polygon_segmentation_with_gcn.src.dataset import PolygonGraphDataset
from polygon_segmentation_with_gcn.src.data_creator import DataCreatorHelper


class GeometricLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @staticmethod
    @ray.remote
    def compute_geometric_loss(each_data: Data, each_segmentation_indices: torch.Tensor) -> float:
        """Compute geometric loss for each graph

        Args:
            each_data (Data): Data object of a graph
            each_segmentation_indices (torch.Tensor): Segmentation indices of a graph

        Returns:
            float: Geometric loss
        """

        each_polygon = geometry.Polygon(each_data.x[:, :2].tolist())

        predicted_edges = DataCreatorHelper.connect_polygon_segments_by_indices(
            each_polygon, each_segmentation_indices.detach().cpu().numpy()
        )

        predicted_edges = [e for e in predicted_edges if e.length > 0]

        predicted_edges = [
            DataCreatorHelper.extend_linestring(
                e, -Configuration.LINESTRING_REDUCTION_LENGTH, -Configuration.LINESTRING_REDUCTION_LENGTH
            )
            for e in predicted_edges
        ]

        label_edges = DataCreatorHelper.connect_polygon_segments_by_indices(
            each_polygon, each_data.edge_label_index_only.unique(dim=1).detach().cpu().numpy()
        )

        label_edges = [e for e in label_edges if e.length > 0]

        label_edges = [
            DataCreatorHelper.extend_linestring(
                e, -Configuration.LINESTRING_REDUCTION_LENGTH, -Configuration.LINESTRING_REDUCTION_LENGTH
            )
            for e in label_edges
        ]

        predicted_edges_union = geometry.MultiLineString(predicted_edges)
        label_edges_union = geometry.MultiLineString(label_edges)

        predicted_edges_union_buffered = predicted_edges_union.buffer(
            Configuration.EDGES_BUFFER_DISTANCE, join_style=geometry.JOIN_STYLE.mitre
        )

        label_edge_union_buffered = label_edges_union.buffer(
            Configuration.EDGES_BUFFER_DISTANCE, join_style=geometry.JOIN_STYLE.mitre
        )

        intersection = label_edge_union_buffered.intersection(predicted_edges_union_buffered)

        geometric_loss = intersection.area**2
        geometric_loss *= -1

        return geometric_loss

    def forward(self, data: Batch, infered: List[torch.Tensor]) -> float:
        """Compute geometric loss for graph batch using ray

        Args:
            data (Batch): Batch object of a graph
            infered (List[torch.Tensor]): List of infered segmentation indices by segmenter

        Returns:
            float: Geometric loss
        """

        tasks = [
            self.compute_geometric_loss.remote(data[i].to("cpu"), infered[i].to("cpu")) for i in range(data.num_graphs)
        ]

        all_geometric_losses = ray.get(tasks)

        all_geometric_losses = torch.tensor(all_geometric_losses)
        geometric_loss = all_geometric_losses.sum() / all_geometric_losses.shape[0]

        return geometric_loss


class PolygonSegmenter(nn.Module):
    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        encoder_activation: nn.Module,
        decoder_activation: nn.Module,
        predictor_activation: nn.Module,
        use_skip_connection: bool = Configuration.USE_SKIP_CONNECTION,
    ):
        super().__init__()

        if conv_type == Configuration.GCNCONV:
            conv = GCNConv
        elif conv_type == Configuration.GRAPHCONV:
            conv = GraphConv
        elif conv_type == Configuration.GATCONV:
            conv = GATConv
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}")

        encoder_modules = []
        encoder_modules += [
            (conv(in_channels, hidden_channels), f"{Configuration.INPUT_ARGS} -> x"),
            nn.BatchNorm1d(hidden_channels),
            encoder_activation,
            nn.Dropout(Configuration.DECODER_DROPOUT_RATE),
        ]

        encoder_modules += [
            (conv(hidden_channels, hidden_channels), f"{Configuration.INPUT_ARGS} -> x"),
            nn.BatchNorm1d(hidden_channels),
            encoder_activation,
            nn.Dropout(Configuration.DECODER_DROPOUT_RATE),
        ] * (Configuration.NUM_ENCODER_LAYERS - 2)

        encoder_modules += [(conv(hidden_channels, out_channels), f"{Configuration.INPUT_ARGS} -> x")]

        decoder_in_channels = out_channels * 2
        if use_skip_connection:
            decoder_in_channels += in_channels * 2

        decoder_modules = []
        decoder_modules += [
            nn.Linear(decoder_in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            decoder_activation,
            nn.Dropout(Configuration.DECODER_DROPOUT_RATE),
        ]

        decoder_modules += [
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            decoder_activation,
            nn.Dropout(Configuration.DECODER_DROPOUT_RATE),
        ] * (Configuration.NUM_DECODER_LAYERS - 2)

        decoder_modules += [
            nn.Linear(out_channels, 1),
            nn.Sigmoid(),
        ]

        predictor_modules = []
        predictor_modules += [
            nn.Linear(out_channels, out_channels),
            predictor_activation,
        ]

        predictor_modules += [
            nn.Linear(out_channels, out_channels),
            predictor_activation,
        ] * (Configuration.NUM_PREDICTOR_LAYERS - 2)

        predictor_modules += [
            nn.Linear(out_channels, 3),
            nn.Softmax(dim=1),
        ]

        self.encoder = Sequential(input_args=Configuration.INPUT_ARGS, modules=encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.k_predictor = nn.Sequential(*predictor_modules)

        self.use_skip_connection = use_skip_connection
        self.out_channels = out_channels

        self.conv_type = conv_type
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.predictor_activation = predictor_activation

        self.to(Configuration.DEVICE)

    def encode(self, data: Batch) -> torch.Tensor:
        """Encode the features of polygon graphs

        Args:
            data (Batch): graph batch

        Returns:
            torch.Tensor: Encoded features
        """

        encoded = self.encoder(data.x, data.edge_index, edge_weight=data.edge_weight)

        return encoded

    def decode(self, data: Batch, encoded: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Decode the encoded features of the nodes to predict whether the edges are connected

        Args:
            data (Batch): graph batch
            encoded (torch.Tensor): Encoded features
            edge_label_index (torch.Tensor): indices labels

        Returns:
            torch.Tensor: whether the edges are connected
        """

        # Merge raw features and encoded features to inject geometric informations
        if self.use_skip_connection:
            encoded = torch.cat([data.x, encoded], dim=1)

        decoded = self.decoder(torch.cat([encoded[edge_label_index[0]], encoded[edge_label_index[1]]], dim=1)).squeeze()

        return decoded

    def _train_predictor(
        self, predictor: nn.Module, each_graph: Data, each_encoded_features: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Train the predictor model to predict k

        Args:
            predictor (nn.Module): predictor model
            each_graph (Data): Data object of a graph
            each_encoded_features (torch.Tensor): Encoded features

        Returns:
            Tuple[torch.Tensor, int]: Predicted k and target k
        """

        unique_edge_index = each_graph.edge_label_index_only.unique(dim=1).t().tolist()

        k_predicted = predictor(each_encoded_features.mean(dim=0).unsqueeze(dim=0))
        k_target = len(unique_edge_index)

        return k_predicted, k_target

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward method of the models, segmenter and predictor

        Args:
            data (Batch): graph batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: whether the edges are connected, predicted k and target k
        """

        # Encode the features of polygon graphs
        encoded = self.encode(data)

        cumulative_num_nodes = 0

        # Train the predictor model to predict k
        k_pred_target_list = []
        for gi in range(data.num_graphs):
            each_graph = data[gi]

            start = cumulative_num_nodes
            end = cumulative_num_nodes + each_graph.num_nodes

            each_encoded_features = encoded[start:end, :]

            cumulative_num_nodes += each_graph.num_nodes

            assert each_encoded_features.shape == (each_graph.num_nodes, self.out_channels)

            k_pred_target_list.append(self._train_predictor(self.k_predictor, each_graph, each_encoded_features))

        k_predicted_list = list(map(lambda x: x[0], k_pred_target_list))
        k_target_list = list(map(lambda x: x[1], k_pred_target_list))

        assert len(k_predicted_list) == len(k_target_list)

        k_predictions = torch.cat(k_predicted_list, dim=0).to(Configuration.DEVICE)
        k_targets = torch.tensor(k_target_list).to(Configuration.DEVICE)

        # Sample negative edges
        negative_edge_index = negative_sampling(
            edge_index=data.edge_label_index_only,
            num_nodes=data.num_nodes,
            num_neg_samples=int(data.edge_label_index_only.shape[1] * Configuration.NEGATIVE_SAMPLE_MULTIPLIER),
            method="sparse",
        )

        # Decode the encoded features of the nodes to predict whether the edges are connected
        decoded = self.decode(data, encoded, torch.hstack([data.edge_label_index_only, negative_edge_index]).int())

        return decoded, k_predictions, k_targets

    @torch.no_grad()
    def infer(self, data_to_infer: Batch, use_filtering: bool = True) -> List[torch.Tensor]:
        """Infer the segmentation of each graph

        Args:
            data_to_infer (Batch): graph batch
            use_filtering (bool, optional): whether to use filtering through rules. Defaults to True.

        Returns:
            List[torch.Tensor]: List of infered segmentation indices
        """

        self.eval()

        infered = []

        for di in range(len(data_to_infer)):
            each_data = data_to_infer[di]

            node_pairs = torch.combinations(torch.arange(each_data.num_nodes), r=2).to(Configuration.DEVICE)

            # Filter existing node pairs from the all node pairs
            node_pairs_filtered = [[], []]
            for node_pair in node_pairs:
                if node_pair.tolist() in each_data.edge_index.t().tolist():
                    continue
                elif node_pair.tolist()[::-1] in each_data.edge_index.t().tolist():
                    continue

                node_pairs_filtered[0].append(node_pair[0].item())
                node_pairs_filtered[1].append(node_pair[1].item())

            node_pairs = torch.tensor(node_pairs_filtered).to(Configuration.DEVICE)

            # Encode the features of a polygon graph
            encoded = self.encode(each_data)

            # Decode the encoded features to obtain the connection probabilities for each node pair
            connection_probabilities = self.decode(each_data, encoded, node_pairs)
            connection_probabilities = torch.where(
                connection_probabilities < Configuration.CONNECTION_THRESHOLD,
                torch.zeros_like(connection_probabilities),
                connection_probabilities,
            )

            # Predict the number of segments
            predicted_k = self.k_predictor(encoded.mean(dim=0).unsqueeze(dim=0)).argmax().item()
            top_10_indices = torch.topk(connection_probabilities, k=Configuration.TOPK).indices

            connected_pairs = node_pairs.t()[top_10_indices].t()

            # Filter bad predictions
            if use_filtering:
                each_polygon = geometry.Polygon(each_data.x[:, :2].tolist())

                filtered_pairs = [[], []]
                for pair in connected_pairs.t():
                    if len(filtered_pairs[0]) == predicted_k:
                        break

                    if bool(pair[0] == pair[1]):
                        continue
                    if pair.tolist() in each_data.edge_index.t().tolist():
                        continue
                    elif pair.tolist()[::-1] in each_data.edge_index.t().tolist():
                        continue

                    segment, *_ = DataCreatorHelper.connect_polygon_segments_by_indices(
                        each_polygon, pair.unsqueeze(1).detach().cpu().numpy()
                    )

                    if segment.length < Configuration.LINESTRING_REDUCTION_LENGTH * 2:
                        continue

                    reduced_segment = DataCreatorHelper.extend_linestring(
                        segment, -Configuration.LINESTRING_REDUCTION_LENGTH, -Configuration.LINESTRING_REDUCTION_LENGTH
                    )

                    if not reduced_segment.within(each_polygon):
                        continue
                    if reduced_segment.within(
                        each_polygon.exterior.buffer(Configuration.POLYGON_EXTERIOR_BUFFER_DISTANCE)
                    ):
                        continue

                    exterior_with_segment = ops.unary_union([segment] + DataCreatorHelper.explode_polygon(each_polygon))

                    exterior_with_segment = shapely.set_precision(
                        exterior_with_segment, Configuration.TOLERANCE_LARGE, mode="valid_output"
                    )

                    segmented = list(ops.polygonize(exterior_with_segment))

                    if len(segmented) != predicted_k + 1:
                        continue

                    segment_coords = np.array(exterior_with_segment.geoms[0].coords)

                    is_satisfied = True
                    for segmented_part in segmented:
                        if segmented_part.area < each_polygon.area * Configuration.AREA_THRESHOLD:
                            is_satisfied = False
                            break

                        segmented_part_degrees = DataCreatorHelper.compute_polyon_inner_degrees(segmented_part)
                        start_index = np.isclose(np.array(segmented_part.exterior.coords), segment_coords[0]).nonzero()[
                            0
                        ][0]
                        end_index = np.isclose(np.array(segmented_part.exterior.coords), segment_coords[1]).nonzero()[
                            0
                        ][0]

                        if segmented_part_degrees[start_index] < Configuration.DEGREE_THRESHOLD:
                            is_satisfied = False
                            break

                        if segmented_part_degrees[end_index] < Configuration.DEGREE_THRESHOLD:
                            is_satisfied = False
                            break

                    if not is_satisfied:
                        continue

                    filtered_pairs[0].append(pair[0].item())
                    filtered_pairs[1].append(pair[1].item())

                connected_pairs = torch.tensor(filtered_pairs)[:, :predicted_k].to(Configuration.DEVICE)

            else:
                connected_pairs = connected_pairs[:, :predicted_k]

            infered.append(connected_pairs)

        self.train()

        return infered


class PolygonSegmenterTrainer:
    def __init__(
        self,
        dataset: PolygonGraphDataset,
        model: nn.Module,
        pre_trained_path: str = None,
        is_debug_mode: bool = False,
        use_geometric_loss: bool = False,
        use_label_smoothing: bool = False,
    ):
        self.dataset = dataset
        self.model = model
        self.pre_trained_path = pre_trained_path
        self.is_debug_mode = is_debug_mode
        self.use_geometric_loss = use_geometric_loss
        self.use_label_smoothing = use_label_smoothing

        if self.is_debug_mode:
            add_debugvisualizer(globals())

        self._set_summary_writer()
        self._set_loss_function()
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_metrics()

        if self.states:
            self.model.load_state_dict(self.states["segmenter_state_dict"])
            self.model.k_predictor.load_state_dict(self.states["predictor_state_dict"])
            self.segmenter_optimizer.load_state_dict(self.states["segmenter_optimizer_state_dict"])
            self.segmenter_lr_scheduler.load_state_dict(self.states["segmenter_lr_scheduler_state_dict"])
            self.predictor_optimizer.load_state_dict(self.states["predictor_optimizer_state_dict"])
            self.predictor_lr_scheduler.load_state_dict(self.states["predictor_lr_scheduler_state_dict"])
            print(f"Set pre-trained all states from {self.states_path} \n")

    def _set_summary_writer(self) -> None:
        """Set the summary writer to record and reproduce the training process"""

        self.log_dir = os.path.join(Configuration.LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.pre_trained_path is not None:
            self.log_dir = self.pre_trained_path

        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        self.summary_writer.add_text("conv_type", self.model.conv_type)
        self.summary_writer.add_text("encoder_activation", str(self.model.encoder_activation))
        self.summary_writer.add_text("decoder_activation", str(self.model.decoder_activation))
        self.summary_writer.add_text("predictor_activation", str(self.model.predictor_activation))

        for key, value in Configuration():
            self.summary_writer.add_text(key, str(value))

        self.states_dir = os.path.join(self.log_dir, "states")
        os.makedirs(self.states_dir, exist_ok=True)

        self.states = {}
        self.states_path = os.path.join(self.states_dir, Configuration.STATES_PTH)
        if len(os.listdir(self.states_dir)) > 0:
            self.states = torch.load(self.states_path)
            print(f"Load pre-trained states from {self.states_path} \n")

    def _set_loss_function(self) -> None:
        """Set all loss functions"""

        self.segmenter_loss_function = nn.BCELoss(
            weight=torch.tensor(Configuration.BCE_LOSS_WEIGHT).to(Configuration.DEVICE),
        )

        self.predictor_loss_function = nn.CrossEntropyLoss(weight=None, reduction="none")

        self.geometric_loss_function = GeometricLoss(weight=Configuration.GEOMETRIC_LOSS_WEIGHT)

    def _set_optimizer(self) -> None:
        """Set all optimizers"""

        self.segmenter_optimizer = torch.optim.Adam(self.model.parameters(), lr=Configuration.SEGMENTER_LEARNING_RATE)
        self.predictor_optimizer = torch.optim.Adam(
            self.model.k_predictor.parameters(), lr=Configuration.PREDICTOR_LEARNING_RATE
        )

    def _set_lr_scheduler(self) -> None:
        """Set all learning rate schedulers"""

        self.segmenter_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.segmenter_optimizer, patience=5, verbose=True, factor=0.1
        )

        self.predictor_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.predictor_optimizer, patience=5, verbose=True, factor=0.1
        )

    def _set_metrics(self) -> None:
        """Set all metrics to evaluate the model"""

        self.accuracy_metric = Accuracy(task="binary").to(Configuration.DEVICE)
        self.f1_score_metric = F1Score(task="binary").to(Configuration.DEVICE)
        self.auroc_metric = AUROC(task="binary").to(Configuration.DEVICE)
        self.recall_metric = Recall(task="binary").to(Configuration.DEVICE)

    def _get_labels(self, data: Batch, use_label_smoothing: bool) -> torch.Tensor:
        """Generate labels for the positive and negative edges

        Args:
            data (Batch): graph batch
            use_label_smoothing (bool): whether to use label smoothing

        Returns:
            torch.Tensor: labels for the positive and negative edges
        """

        ones = torch.ones(data.edge_label_index_only.shape[1])
        zeros = torch.tensor([0] * int(data.edge_label_index_only.shape[1] * Configuration.NEGATIVE_SAMPLE_MULTIPLIER))

        labels = torch.hstack([ones, zeros]).to(Configuration.DEVICE)

        if use_label_smoothing:
            labels = (1 - Configuration.LABEL_SMOOTHING_FACTOR) * labels + Configuration.LABEL_SMOOTHING_FACTOR / 2

        return labels

    def _compute_focal_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss for the predictor model

        Args:
            loss (torch.Tensor): loss for the predictor model

        Returns:
            torch.Tensor: loss
        """

        pt = torch.exp(-loss)
        focal_loss = Configuration.FOCAL_LOSS_ALPHA * (1 - pt) ** Configuration.FOCAL_LOSS_GAMMA * loss

        return focal_loss.mean()

    @runtime_calculator
    def _train_each_epoch(
        self,
        dataset: PolygonGraphDataset,
        model: nn.Module,
        segmenter_loss_function: nn.modules.loss._Loss,
        predictor_loss_function: nn.modules.loss._Loss,
        geometric_loss_function: GeometricLoss,
        segmenter_optimizer: torch.optim.Optimizer,
        predictor_optimizer: torch.optim.Optimizer,
        use_geometric_loss: bool,
        use_label_smoothing: bool,
        epoch: int,
    ) -> Tuple[float]:
        """Train the models for each epoch

        Args:
            dataset (PolygonGraphDataset): dataset
            model (nn.Module): model
            segmenter_loss_function (nn.modules.loss._Loss): segmenter loss function
            predictor_loss_function (nn.modules.loss._Loss): predictor loss function
            geometric_loss_function (GeometricLoss): geometric loss function
            segmenter_optimizer (torch.optim.Optimizer): segmenter optimizer
            predictor_optimizer (torch.optim.Optimizer): predictor optimizer
            use_geometric_loss (bool): whether to use geometric loss
            use_label_smoothing (bool): whether to use label smoothing
            epoch (int): each epoch

        Returns:
            Tuple[float]: average loss for the train
        """

        train_losses = []
        for data_to_train in tqdm(dataset.train_dataloader, desc=f"Training... epoch: {epoch}/{Configuration.EPOCH}"):
            train_decoded, train_k_predictions, train_k_targets = model(data_to_train)

            train_labels = self._get_labels(data_to_train, use_label_smoothing)

            train_segmenter_loss = segmenter_loss_function(train_decoded, train_labels)

            train_geometric_loss = 0
            if use_geometric_loss:
                train_infered = model.infer(data_to_train, use_filtering=False)
                train_geometric_loss = geometric_loss_function(data_to_train, train_infered).item()

            train_predictor_loss = predictor_loss_function(train_k_predictions, train_k_targets)
            train_predictor_loss = self._compute_focal_loss(train_predictor_loss)

            train_total_loss = train_segmenter_loss + train_predictor_loss + train_geometric_loss

            train_losses.append(train_total_loss.item())

            model.zero_grad()
            train_total_loss.backward()
            segmenter_optimizer.step()
            predictor_optimizer.step()

        train_loss_avg = sum(train_losses) / len(train_losses)

        return train_loss_avg

    @torch.no_grad()
    def _evaluate_each_epoch(
        self,
        dataset: PolygonGraphDataset,
        model: nn.Module,
        segmenter_loss_function: nn.modules.loss._Loss,
        predictor_loss_function: nn.modules.loss._Loss,
        geometric_loss_function: GeometricLoss,
        use_geometric_loss: bool,
        use_label_smoothing: bool,
        accuracy_metric: Accuracy,
        f1_score_metric: F1Score,
        auroc_metric: AUROC,
        recall_metric: Recall,
        epoch: int,
        is_test: bool = False,
    ) -> Tuple[float]:
        """Evaluate the models for each epoch

        Args:
            dataset (PolygonGraphDataset): dataset
            model (nn.Module): model
            segmenter_loss_function (nn.modules.loss._Loss): segmenter loss function
            predictor_loss_function (nn.modules.loss._Loss): predictor loss function
            geometric_loss_function (GeometricLoss): geometric loss function
            use_geometric_loss (bool): whether to use geometric loss
            use_label_smoothing (bool): whether to use label smoothing
            accuracy_metric (Accuracy): accuracy metric
            f1_score_metric (F1Score): f1 score metric
            auroc_metric (AUROC): auroc metric
            recall_metric (Recall): recall metric
            epoch (int): each epoch
            is_test (bool, optional): whether to evaluate on the test dataset. Defaults to False.

        Returns:
            Tuple[float]: average loss for the validation
        """

        model.eval()
        accuracy_metric.reset()
        f1_score_metric.reset()
        auroc_metric.reset()
        recall_metric.reset()

        dataloader_to_validate = dataset.validation_dataloader
        if is_test:
            dataloader_to_validate = dataset.test_dataloader

        validation_losses = []
        for data_to_validate in tqdm(
            dataloader_to_validate, desc=f"Evaluating... epoch: {epoch}/{Configuration.EPOCH}"
        ):
            validation_decoded, validation_k_predictions, validation_k_targets = model(data_to_validate)

            validation_labels = self._get_labels(data_to_validate, use_label_smoothing)

            validation_segmenter_loss = segmenter_loss_function(validation_decoded, validation_labels)

            validation_geometric_loss = 0
            if use_geometric_loss:
                validation_infered = model.infer(data_to_validate, use_filtering=False)
                validation_geometric_loss = geometric_loss_function(data_to_validate, validation_infered).item()

            validation_predictor_loss = predictor_loss_function(validation_k_predictions, validation_k_targets)
            validation_predictor_loss = self._compute_focal_loss(validation_predictor_loss)

            validation_total_loss = validation_segmenter_loss + validation_predictor_loss + validation_geometric_loss

            validation_labels_without_smoothing = torch.where(
                torch.isclose(validation_labels, torch.tensor(1 - Configuration.LABEL_SMOOTHING_FACTOR / 2)),
                torch.tensor(1).to(Configuration.DEVICE),
                torch.tensor(0).to(Configuration.DEVICE),
            )

            accuracy_metric.update(
                (validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(),
                validation_labels_without_smoothing,
            )
            f1_score_metric.update(
                (validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(),
                validation_labels_without_smoothing,
            )
            auroc_metric.update(
                (validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(),
                validation_labels_without_smoothing,
            )
            recall_metric.update(
                (validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(),
                validation_labels_without_smoothing,
            )

            validation_losses.append(validation_total_loss.item())

        model.train()

        validation_loss_avg = sum(validation_losses) / len(validation_losses)

        validation_accuracy = accuracy_metric.compute().item()
        validation_f1_score = f1_score_metric.compute().item()
        validation_auroc = auroc_metric.compute().item()
        validation_recall = recall_metric.compute().item()

        return validation_loss_avg, validation_accuracy, validation_f1_score, validation_auroc, validation_recall

    def _get_figures_to_evaluate_qualitatively(self, batch: Batch, indices: List[torch.Tensor]) -> List[plt.Figure]:
        """Get figures to evaluate qualitatively

        Args:
            batch (Batch): batch
            indices (List[torch.Tensor]): indices

        Returns:
            List[plt.Figure]: figures
        """

        figures = []

        for si in range(len(indices)):
            each_data = batch[si]
            each_segmentation_indices = indices[si]

            polygon = geometry.Polygon(each_data.x[:, :2].detach().cpu().numpy())
            predicted_edges = DataCreatorHelper.connect_polygon_segments_by_indices(
                polygon, each_segmentation_indices.detach().cpu().numpy()
            )

            predicted_edges_zip = list(zip(predicted_edges, each_segmentation_indices.detach().cpu().numpy().T))

            label_edges = DataCreatorHelper.connect_polygon_segments_by_indices(
                polygon, each_data.edge_label_index_only.T.unique(dim=0).detach().cpu().numpy().T
            )

            figure = plt.figure(figsize=Configuration.FIGSIZE, dpi=Configuration.DPI)
            ax = figure.add_subplot(1, 1, 1)
            ax.axis("equal")

            added_ground_truth_label = False
            for label_edge in label_edges:
                if added_ground_truth_label:
                    ax.plot(*label_edge.coords.xy, color="blue", linewidth=1.0, alpha=0.3)
                else:
                    added_ground_truth_label = True
                    ax.plot(*label_edge.coords.xy, color="blue", linewidth=1.0, alpha=0.3, label="ground truth")

            added_predicted_label = False
            for predicted_edge, predicted_index in predicted_edges_zip:
                if added_predicted_label:
                    ax.plot(*predicted_edge.coords.xy, color="green", linewidth=1.0)
                else:
                    added_predicted_label = True
                    ax.plot(
                        *predicted_edge.coords.xy,
                        color="green",
                        linewidth=1.0,
                        label="predicted",
                    )

                if predicted_index.tolist() in each_data.edge_label_index_only.T.tolist():
                    ax.plot(*predicted_edge.coords.xy, color="yellow", linewidth=1.0)
                elif predicted_index.tolist()[::-1] in each_data.edge_label_index_only.T.tolist():
                    ax.plot(*predicted_edge.coords.xy, color="yellow", linewidth=1.0)

            ax.plot(*polygon.exterior.coords.xy, color="black", linewidth=0.6, alpha=0.3, label="polygon")
            ax.fill(*polygon.exterior.coords.xy, alpha=0.1, color="black")

            x, y = polygon.exterior.coords.xy
            ax.scatter(x, y, color="red", s=7, label="vertices")
            ax.grid(True, color="lightgray")

            annotation = f"""
                loc: {int(each_data.loc)}
                name: {each_data.name}
                predicted k : {each_segmentation_indices.shape[1]}
            """

            plt.axis([-2.0, 2.0, -2.0, 2.0])

            plt.gcf().text(
                0.45,
                0.2,
                annotation,
                va="center",
                ha="center",
                color="black",
                fontsize=8,
            )

            plt.legend(fontsize="small")

            figures.append(figure)

            plt.close(figure)

        return figures

    def _merge_figures(self, figures: List[plt.Figure]) -> Image:
        """Merge figures

        Args:
            viz_count (int): the number of figures to merge
            figures (List[plt.Figure]): figures to merge

        Returns:
            Image: merged image
        """

        col_num = 5
        row_num = int(np.ceil(len(figures) / col_num))
        img_size = Configuration.FIGSIZE[0] * Configuration.DPI
        merged_image = Image.new("RGB", (col_num * img_size, row_num * img_size), "white")

        current_cols = 0
        output_height = 0
        output_width = 0

        for figure in figures:
            image = DataCreatorHelper.fig_to_img(figure)

            merged_image.paste(image, (output_width, output_height))

            current_cols += 1
            if current_cols >= col_num:
                output_width = 0
                output_height += img_size
                current_cols = 0
            else:
                output_width += img_size

        return merged_image

    @runtime_calculator
    def evaluate_qualitatively(
        self, dataset: PolygonGraphDataset, model: nn.Module, epoch: int, viz_count: int = 10
    ) -> None:
        """Evaluate qualitatively by visualizing the segmentation results

        Args:
            dataset (PolygonGraphDataset): dataset
            model (nn.Module): model
            epoch (int): each epoch
            viz_count (int, optional): the number of data to visualize. Defaults to 10.
        """

        irregular_train_indices_to_viz = torch.randperm(len(dataset.train_dataset.datasets[1]))[:viz_count]
        irregular_train_subset = Subset(dataset.train_dataset.datasets[1], irregular_train_indices_to_viz)
        irregular_train_sampled = DataLoader(irregular_train_subset, batch_size=viz_count)

        regular_train_indices_to_viz = torch.randperm(len(dataset.train_dataset.datasets[0]))[:viz_count]
        regular_train_subset = Subset(dataset.train_dataset.datasets[0], regular_train_indices_to_viz)
        regular_train_sampled = DataLoader(regular_train_subset, batch_size=viz_count)

        irregular_train_batch = [data_to_infer for data_to_infer in irregular_train_sampled][0]
        regular_train_batch = [data_to_infer for data_to_infer in regular_train_sampled][0]

        irregular_train_segmentation_indices = model.infer(irregular_train_batch, use_filtering=True)
        regular_train_segmentation_indices = model.infer(regular_train_batch, use_filtering=True)

        figures = []

        figures += self._get_figures_to_evaluate_qualitatively(
            irregular_train_batch, irregular_train_segmentation_indices
        )
        figures += self._get_figures_to_evaluate_qualitatively(regular_train_batch, regular_train_segmentation_indices)

        irregular_validation_indices_to_viz = torch.randperm(len(dataset.validation_dataset.datasets[1]))[:viz_count]
        irregular_validation_subset = Subset(
            dataset.validation_dataset.datasets[1], irregular_validation_indices_to_viz
        )
        irregular_validation_sampled = DataLoader(irregular_validation_subset, batch_size=viz_count)

        regular_validation_indices_to_viz = torch.randperm(len(dataset.validation_dataset.datasets[0]))[:viz_count]
        regular_validation_subset = Subset(dataset.validation_dataset.datasets[0], regular_validation_indices_to_viz)
        regular_validation_sampled = DataLoader(regular_validation_subset, batch_size=viz_count)

        irregular_validation_batch = [data_to_infer for data_to_infer in irregular_validation_sampled][0]
        regular_validation_batch = [data_to_infer for data_to_infer in regular_validation_sampled][0]

        irregular_validation_segmentation_indices = model.infer(irregular_validation_batch, use_filtering=True)
        regular_validation_segmentation_indices = model.infer(regular_validation_batch, use_filtering=True)

        figures += self._get_figures_to_evaluate_qualitatively(
            irregular_validation_batch, irregular_validation_segmentation_indices
        )
        figures += self._get_figures_to_evaluate_qualitatively(
            regular_validation_batch, regular_validation_segmentation_indices
        )

        merged_image = self._merge_figures(figures)

        self.summary_writer.add_image(
            f"qualitative_evaluation_{epoch}", np.array(merged_image), epoch, dataformats="HWC"
        )

    def train(self) -> None:
        """Train the models"""

        if self.use_geometric_loss:
            ray.init()

        best_validation_loss = torch.inf
        start = 1

        if len(self.states) > 0:
            start = self.states["epoch"] + 1
            best_validation_loss = self.states["best_validation_loss"]

        for epoch in range(start, Configuration.EPOCH + 1):
            train_loss_avg = self._train_each_epoch(
                self.dataset,
                self.model,
                self.segmenter_loss_function,
                self.predictor_loss_function,
                self.geometric_loss_function,
                self.segmenter_optimizer,
                self.predictor_optimizer,
                self.use_geometric_loss,
                self.use_label_smoothing,
                epoch,
            )

            (
                validation_loss_avg,
                validation_accuracy,
                validation_f1_score,
                validation_auroc,
                validation_recall,
            ) = self._evaluate_each_epoch(
                self.dataset,
                self.model,
                self.segmenter_loss_function,
                self.predictor_loss_function,
                self.geometric_loss_function,
                self.use_geometric_loss,
                self.use_label_smoothing,
                self.accuracy_metric,
                self.f1_score_metric,
                self.auroc_metric,
                self.recall_metric,
                epoch,
            )

            self.segmenter_lr_scheduler.step(validation_loss_avg)

            if validation_loss_avg < best_validation_loss:
                print(
                    f"""saving model...
                        existing loss: {best_validation_loss}
                        new loss: {validation_loss_avg}
                    """
                )

                best_validation_loss = validation_loss_avg

                states = {
                    "epoch": epoch,
                    "best_validation_loss": best_validation_loss,
                    "segmenter_state_dict": self.model.state_dict(),
                    "predictor_state_dict": self.model.k_predictor.state_dict(),
                    "segmenter_optimizer_state_dict": self.segmenter_optimizer.state_dict(),
                    "segmenter_lr_scheduler_state_dict": self.segmenter_lr_scheduler.state_dict(),
                    "predictor_optimizer_state_dict": self.predictor_optimizer.state_dict(),
                    "predictor_lr_scheduler_state_dict": self.predictor_lr_scheduler.state_dict(),
                    "configuration": {k: v for k, v in Configuration()},
                }

                torch.save(states, self.states_path)

            else:
                states = torch.load(self.states_path)
                states.update({"epoch": epoch})

                torch.save(states, self.states_path)

            self.summary_writer.add_scalar("segmenter_train_loss", train_loss_avg, epoch)
            self.summary_writer.add_scalar("segmenter_validation_loss", validation_loss_avg, epoch)
            self.summary_writer.add_scalar("segmenter_validation_accuracy", validation_accuracy, epoch)
            self.summary_writer.add_scalar("segmenter_validation_f1_score", validation_f1_score, epoch)
            self.summary_writer.add_scalar("segmenter_validation_auroc", validation_auroc, epoch)
            self.summary_writer.add_scalar("segmenter_validation_recall", validation_recall, epoch)

            self.evaluate_qualitatively(self.dataset, self.model, epoch)

            print(
                f"""training status
                    Epoch: {epoch}th
                    Average Train Loss: {train_loss_avg}
                    Average Validation Loss: {validation_loss_avg}
                    Validation Accuracy: {validation_accuracy}
                    Validation F1 Score: {validation_f1_score}
                    Validation AUROC: {validation_auroc}
                    Validation Recall: {validation_recall}
                """
            )

    def test(self) -> None:
        """Test segmenter model"""

        (
            test_loss_avg,
            test_accuracy,
            test_f1_score,
            test_auroc,
            test_recall,
        ) = self._evaluate_each_epoch(
            self.dataset,
            self.model,
            self.segmenter_loss_function,
            self.predictor_loss_function,
            self.geometric_loss_function,
            self.use_geometric_loss,
            self.use_label_smoothing,
            self.accuracy_metric,
            self.f1_score_metric,
            self.auroc_metric,
            self.recall_metric,
            epoch=None,
            is_test=True,
        )

        self.summary_writer.add_scalar("segmenter_test_loss", test_loss_avg)
        self.summary_writer.add_scalar("segmenter_test_accuracy", test_accuracy)
        self.summary_writer.add_scalar("segmenter_test_f1_score", test_f1_score)
        self.summary_writer.add_scalar("segmenter_test_auroc", test_auroc)
        self.summary_writer.add_scalar("segmenter_test_recall", test_recall)

        print(
            f"""test status
                Average Test Loss: {test_loss_avg}
                Test Accuracy: {test_accuracy}
                Test F1 Score: {test_f1_score}
                Test AUROC: {test_auroc}
                Test Recall: {test_recall}
            """
        )

        viz_count = 50
        g = torch.Generator()

        regular_test_indices_to_viz = torch.randperm(len(self.dataset.test_dataset.datasets[0]), generator=g)[
            :viz_count
        ]
        regular_test_subset = Subset(self.dataset.test_dataset.datasets[0], regular_test_indices_to_viz)
        regular_test_sampled = DataLoader(regular_test_subset, batch_size=viz_count)

        irregular_test_indices_to_viz = torch.randperm(len(self.dataset.test_dataset.datasets[1]), generator=g)[
            :viz_count
        ]
        irregular_test_subset = Subset(self.dataset.test_dataset.datasets[1], irregular_test_indices_to_viz)
        irregular_test_sampled = DataLoader(irregular_test_subset, batch_size=viz_count)

        irregular_test_batch = [data_to_infer for data_to_infer in irregular_test_sampled][0]
        regular_test_batch = [data_to_infer for data_to_infer in regular_test_sampled][0]

        irregular_test_segmentation_indices = self.model.infer(irregular_test_batch, use_filtering=True)
        regular_test_segmentation_indices = self.model.infer(regular_test_batch, use_filtering=True)

        figures = []

        figures += self._get_figures_to_evaluate_qualitatively(
            irregular_test_batch, irregular_test_segmentation_indices
        )
        figures += self._get_figures_to_evaluate_qualitatively(regular_test_batch, regular_test_segmentation_indices)

        merged_image = self._merge_figures(figures)

        self.summary_writer.add_image("qualitative_evaluation_test", np.array(merged_image), 0, dataformats="HWC")

        self.model.train()
