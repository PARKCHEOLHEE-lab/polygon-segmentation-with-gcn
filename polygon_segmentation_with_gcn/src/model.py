import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import torch
import torch.nn as nn
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ray

from PIL import Image
from shapely import geometry
from typing import List, Tuple
from tqdm import tqdm
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv, GraphConv, GATConv
from torch_geometric.utils import negative_sampling
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score, AUROC

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
    def compute_geometric_loss(each_data: Data, each_segmentation_indices: torch.Tensor):
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
        use_skip_connection: bool = True,
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

        self.to(Configuration.DEVICE)

    def encode(self, data: Batch) -> torch.Tensor:
        """_summary_

        Args:
            data (Batch): _description_

        Returns:
            torch.Tensor: _description_
        """

        encoded = self.encoder(data.x, data.edge_index, edge_weight=data.edge_weight)

        return encoded

    def decode(self, data: Batch, encoded: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            encoded (torch.Tensor): _description_
            edge_label_index (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # Merge raw features and encoded features to inject geometric informations
        if self.use_skip_connection:
            encoded = torch.cat([data.x, encoded], dim=1)

        decoded = self.decoder(torch.cat([encoded[edge_label_index[0]], encoded[edge_label_index[1]]], dim=1)).squeeze()

        return decoded

    @staticmethod
    def _train_predictor(predictor, each_graph, each_encoded_features):
        unique_edge_index = each_graph.edge_label_index_only.unique(dim=1).t().tolist()

        k_predicted = predictor(each_encoded_features.mean(dim=0).unsqueeze(dim=0))
        k_target = 2
        if [0, 1] in unique_edge_index:
            k_target -= 1
        if [1, 2] in unique_edge_index:
            k_target -= 1

        return k_predicted, k_target

    def forward(self, data: Batch) -> torch.Tensor:
        # Encode the features of polygon graphs
        encoded = self.encode(data)

        cumulative_num_nodes = 0

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
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=int(data.edge_label_index_only.shape[1] * Configuration.NEGATIVE_SAMPLE_MULTIPLIER),
            method="sparse",
        )

        # Decode the encoded features of the nodes to predict whether the edges are connected
        decoded = self.decode(data, encoded, torch.hstack([data.edge_label_index_only, negative_edge_index]).int())

        return decoded, k_predictions, k_targets

    @torch.no_grad()
    def infer(self, data_to_infer: Batch, use_filtering: bool = True) -> List[torch.Tensor]:
        self.eval()

        infered = []

        for di in range(len(data_to_infer)):
            each_data = data_to_infer[di]

            node_pairs = torch.combinations(torch.arange(each_data.num_nodes), r=2).to(Configuration.DEVICE)

            node_pairs_filtered = [[], []]
            for node_pair in node_pairs:
                if node_pair.tolist() in each_data.edge_index.t().tolist():
                    continue
                elif node_pair.tolist()[::-1] in each_data.edge_index.t().tolist():
                    continue

                node_pairs_filtered[0].append(node_pair[0].item())
                node_pairs_filtered[1].append(node_pair[1].item())

            node_pairs = torch.tensor(node_pairs_filtered).to(Configuration.DEVICE)

            encoded = self.encode(each_data)
            connection_probabilities = self.decode(each_data, encoded, node_pairs)
            connection_probabilities = torch.where(
                connection_probabilities < Configuration.CONNECTION_THRESHOLD,
                torch.zeros_like(connection_probabilities),
                connection_probabilities,
            )

            topk_indices = torch.topk(
                connection_probabilities, k=self.k_predictor(encoded.mean(dim=0).unsqueeze(dim=0)).argmax().item()
            ).indices

            connected_pairs = node_pairs.t()[topk_indices].t()

            if use_filtering:
                each_polygon = geometry.Polygon(each_data.x[:, :2].tolist())

                filtered_pairs = [[], []]
                for pair in connected_pairs:
                    if bool(node_pair[0] == node_pair[1]):
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

                    filtered_pairs[0].append(pair[0].item())
                    filtered_pairs[1].append(pair[1].item())

                connected_pairs = torch.tensor(filtered_pairs).to(Configuration.DEVICE)

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
    ):
        self.dataset = dataset
        self.model = model
        self.pre_trained_path = pre_trained_path
        self.is_debug_mode = is_debug_mode
        self.use_geometric_loss = use_geometric_loss

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

    def _set_summary_writer(self):
        self.log_dir = os.path.join(Configuration.LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.pre_trained_path is not None:
            self.log_dir = self.pre_trained_path

        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        for key, value in Configuration():
            self.summary_writer.add_text(key, str(value))

        self.states_dir = os.path.join(self.log_dir, "states")
        os.makedirs(self.states_dir, exist_ok=True)

        self.states = {}
        self.states_path = os.path.join(self.states_dir, Configuration.STATES_PTH)
        if len(os.listdir(self.states_dir)) > 0:
            self.states = torch.load(self.states_path)
            print(f"Load pre-trained states from {self.states_path} \n")

    def _set_loss_function(self):
        self.segmenter_loss_function = nn.BCELoss(
            weight=torch.tensor(Configuration.BCE_LOSS_WEIGHT).to(Configuration.DEVICE)
        )

        self.predictor_loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(Configuration.CROSS_ENTROPY_LOSS_WEIGHT).to(Configuration.DEVICE)
        )

        self.geometric_loss_function = GeometricLoss(weight=Configuration.GEOMETRIC_LOSS_WEIGHT)

    def _set_optimizer(self):
        self.segmenter_optimizer = torch.optim.Adam(self.model.parameters(), lr=Configuration.SEGMENTER_LEARNING_RATE)
        self.predictor_optimizer = torch.optim.Adam(
            self.model.k_predictor.parameters(), lr=Configuration.PREDICTOR_LEARNING_RATE
        )

    def _set_lr_scheduler(self):
        self.segmenter_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.segmenter_optimizer, patience=5, verbose=True, factor=0.1
        )

        self.predictor_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.predictor_optimizer, patience=5, verbose=True, factor=0.1
        )

    def _set_metrics(self):
        self.accuracy_metric = Accuracy(task="binary").to(Configuration.DEVICE)
        self.f1_score_metric = F1Score(task="binary").to(Configuration.DEVICE)
        self.auroc_metric = AUROC(task="binary").to(Configuration.DEVICE)

    def _get_labels(self, data: Batch) -> torch.Tensor:
        ones = torch.ones(data.edge_label_index_only.shape[1])
        zeros = torch.tensor([0] * int(data.edge_label_index_only.shape[1] * Configuration.NEGATIVE_SAMPLE_MULTIPLIER))

        labels = torch.hstack([ones, zeros]).to(Configuration.DEVICE)

        return labels

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
        epoch: int,
    ) -> Tuple[float]:
        """_summary_

        Args:
            dataset (PolygonGraphDataset): _description_
            model (nn.Module): _description_
            segmenter_loss_function (nn.modules.loss._Loss): _description_
            predictor_loss_function (nn.modules.loss._Loss): _description_
            optimizer (torch.optim.Optimizer): _description_
            epoch (int): _description_

        Returns:
            Tuple[float]: _description_
        """

        train_losses = []
        for data_to_train in tqdm(dataset.train_dataloader, desc=f"Training... epoch: {epoch}/{Configuration.EPOCH}"):
            train_decoded, train_k_predictions, train_k_targets = model(data_to_train)

            train_labels = self._get_labels(data_to_train)

            train_segmenter_loss = segmenter_loss_function(train_decoded, train_labels)

            train_geometric_loss = 0
            if use_geometric_loss:
                train_infered = model.infer(data_to_train, use_filtering=False)
                train_geometric_loss = geometric_loss_function(data_to_train, train_infered).item()

            train_predictor_loss = predictor_loss_function(train_k_predictions, train_k_targets)

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
        accuracy_metric: Accuracy,
        f1_score_metric: F1Score,
        auroc_metric: AUROC,
        epoch: int,
    ) -> Tuple[float]:
        """_summary_

        Args:
            dataset (PolygonGraphDataset): _description_
            model (nn.Module): _description_
            segmenter_loss_function (nn.modules.loss._Loss): _description_
            predictor_loss_function (nn.modules.loss._Loss): _description_
            geometric_loss_function (GeometricLoss): _description_
            use_geometric_loss (bool): _description_
            epoch (int): _description_

        Returns:
            Tuple[float]: _description_
        """

        model.eval()
        accuracy_metric.reset()
        f1_score_metric.reset()

        validation_losses = []
        for data_to_validate in tqdm(
            dataset.validation_dataloder, desc=f"Evaluating... epoch: {epoch}/{Configuration.EPOCH}"
        ):
            validation_decoded, validation_k_predictions, validation_k_targets = model(data_to_validate)

            validation_labels = self._get_labels(data_to_validate)

            validation_segmenter_loss = segmenter_loss_function(validation_decoded, validation_labels)

            validation_geometric_loss = 0
            if use_geometric_loss:
                validation_infered = model.infer(data_to_validate, use_filtering=False)
                validation_geometric_loss = geometric_loss_function(data_to_validate, validation_infered).item()

            validation_predictor_loss = predictor_loss_function(validation_k_predictions, validation_k_targets)

            validation_total_loss = validation_segmenter_loss + validation_predictor_loss + validation_geometric_loss

            accuracy_metric.update((validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(), validation_labels)
            f1_score_metric.update((validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(), validation_labels)
            auroc_metric.update((validation_decoded >= Configuration.CONNECTION_THRESHOLD).int(), validation_labels)

            validation_losses.append(validation_total_loss.item())

        model.train()

        validation_loss_avg = sum(validation_losses) / len(validation_losses)

        validation_accuracy = accuracy_metric.compute().item()
        validation_f1_score = f1_score_metric.compute().item()
        validation_auroc = auroc_metric.compute().item()

        return validation_loss_avg, validation_accuracy, validation_f1_score, validation_auroc

    def _get_figures_to_evaluate_qualitatively(self, batch: Batch, indices: List[torch.Tensor]):
        dpi = 100
        figsize = (5, 5)

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

            figure = plt.figure(figsize=figsize, dpi=dpi)
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

            plt.legend()

            figures.append(figure)

            plt.close(figure)

        return figures

    @runtime_calculator
    def evaluate_qualitatively(
        self, dataset: PolygonGraphDataset, model: nn.Module, epoch: int, viz_count: int = 10
    ) -> None:
        """_summary_

        Args:
            dataset (PolygonGraphDataset): _description_
            model (nn.Module): _description_
        """

        irregular_train_indices_to_viz = torch.randperm(len(dataset.train_dataset.datasets[1]))[:viz_count]
        irregular_train_subset = Subset(dataset.train_dataset.datasets[1], irregular_train_indices_to_viz)
        irregular_train_sampled = DataLoader(irregular_train_subset, batch_size=viz_count)

        regular_train_indices_to_viz = torch.randperm(len(dataset.train_dataset.datasets[0]))[:viz_count]
        regular_train_subset = Subset(dataset.train_dataset.datasets[0], regular_train_indices_to_viz)
        regular_train_sampled = DataLoader(regular_train_subset, batch_size=viz_count)

        irregular_train_batch = [data_to_infer for data_to_infer in irregular_train_sampled][0]
        regular_train_batch = [data_to_infer for data_to_infer in regular_train_sampled][0]

        irregular_train_segmentation_indices = model.infer(irregular_train_batch, use_filtering=False)
        regular_train_segmentation_indices = model.infer(regular_train_batch, use_filtering=False)

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

        irregular_validation_segmentation_indices = model.infer(irregular_validation_batch, use_filtering=False)
        regular_validation_segmentation_indices = model.infer(regular_validation_batch, use_filtering=False)

        figures += self._get_figures_to_evaluate_qualitatively(
            irregular_validation_batch, irregular_validation_segmentation_indices
        )
        figures += self._get_figures_to_evaluate_qualitatively(
            regular_validation_batch, regular_validation_segmentation_indices
        )

        dpi = 100
        figsize = (5, 5)

        col_num = 5
        row_num = int(np.ceil((viz_count * 4) / col_num))
        img_size = figsize[0] * dpi
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

        self.summary_writer.add_image(
            f"qualitative_evaluation_{epoch}", np.array(merged_image), epoch, dataformats="HWC"
        )

    def train(self) -> None:
        """_summary_"""

        if self.use_geometric_loss:
            ray.init()

        best_auroc = -torch.inf
        start = 1

        if len(self.states) > 0:
            start = self.states["epoch"] + 1
            best_auroc = self.states["best_auroc"]

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
                epoch,
            )

            validation_loss_avg, validation_accuracy, validation_f1_score, validation_auroc = self._evaluate_each_epoch(
                self.dataset,
                self.model,
                self.segmenter_loss_function,
                self.predictor_loss_function,
                self.geometric_loss_function,
                self.use_geometric_loss,
                self.accuracy_metric,
                self.f1_score_metric,
                self.auroc_metric,
                epoch,
            )

            self.segmenter_lr_scheduler.step(validation_loss_avg)

            if validation_auroc > best_auroc:
                best_loss = validation_loss_avg
                best_auroc = validation_auroc

                states = {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "best_auroc": best_auroc,
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

            self.evaluate_qualitatively(self.dataset, self.model, epoch)

            print(
                f"""
                    Epoch: {epoch}th
                    Average Train Loss: {train_loss_avg}
                    Average Validation Loss: {validation_loss_avg}
                    Validation Accuracy: {validation_accuracy}
                    Validation F1 Score: {validation_f1_score}
                    Validation AUROC: {validation_auroc}
                """
            )


if __name__ == "__main__":
    Configuration.set_seed()

    dataset = PolygonGraphDataset()
    model = PolygonSegmenter(
        conv_type=Configuration.GATCONV,
        in_channels=dataset.regular_polygons[0].x.shape[1],
        hidden_channels=Configuration.HIDDEN_CHANNELS,
        out_channels=Configuration.OUT_CHANNELS,
        encoder_activation=nn.PReLU().to(Configuration.DEVICE),
        decoder_activation=nn.PReLU().to(Configuration.DEVICE),
        predictor_activation=nn.PReLU().to(Configuration.DEVICE),
        use_skip_connection=True,
    )

    polygon_segmenter_trainer = PolygonSegmenterTrainer(
        dataset=dataset,
        model=model,
        is_debug_mode=True,
        pre_trained_path=None,
        use_geometric_loss=False,
    )
    polygon_segmenter_trainer.train()
