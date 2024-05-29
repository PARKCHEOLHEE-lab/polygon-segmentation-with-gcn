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

from PIL import Image
from shapely import geometry
from typing import List
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils import negative_sampling
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from polygon_segmentation_with_gcn.src.commonutils import runtime_calculator, add_debugvisualizer
from polygon_segmentation_with_gcn.src.config import Configuration
from polygon_segmentation_with_gcn.src.dataset import PolygonGraphDataset
from polygon_segmentation_with_gcn.src.data_creator import DataCreatorHelper


class PolygonSegmenterGCNConv(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, activation_function: nn.Module):
        super().__init__()

        input_args = "x, edge_index, edge_weight"

        encoder_modules = []
        encoder_modules.extend(
            [
                (GCNConv(in_channels, hidden_channels), f"{input_args} -> x"),
                nn.BatchNorm1d(hidden_channels),
                activation_function,
                nn.Dropout(Configuration.DROPOUT_RATE),
            ]
        )

        for _ in range(Configuration.NUM_ENCODER_LAYERS - 2):
            encoder_modules.extend(
                [
                    (GCNConv(hidden_channels, hidden_channels), f"{input_args} -> x"),
                    nn.BatchNorm1d(hidden_channels),
                    activation_function,
                    nn.Dropout(Configuration.DROPOUT_RATE),
                ]
            )

        encoder_modules.append((GCNConv(hidden_channels, out_channels), f"{input_args} -> x"))

        self.encoder = Sequential(input_args=input_args, modules=encoder_modules)

        decoder_modules = []
        decoder_modules.extend(
            [
                nn.Linear(out_channels * 2, out_channels),
                activation_function,
            ]
        )

        for _ in range(Configuration.NUM_DECODER_LAYERS - 2):
            decoder_modules.extend(
                [
                    nn.Linear(out_channels, out_channels),
                    activation_function,
                ]
            )

        decoder_modules.append(nn.Linear(out_channels, 1))

        self.decoder = nn.Sequential(*decoder_modules)

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

    def decode(self, encoded: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            encoded (torch.Tensor): _description_
            edge_label_index (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        decoded = self.decoder(torch.cat([encoded[edge_label_index[0]], encoded[edge_label_index[1]]], dim=1)).squeeze()

        return decoded

    def forward(self, data: Batch) -> torch.Tensor:
        # Encode the features of polygon graphs
        encoded = self.encode(data)

        # Sample negative edges
        negative_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=int(data.edge_label_index_only.shape[1] * Configuration.NEGATIVE_SAMPLE_MULTIPLIER),
            method="sparse",
        )

        # Decode the encoded features of the nodes to predict whether the edges are connected
        decoded = self.decode(encoded, edge_label_index=torch.hstack([data.edge_label_index_only, negative_edge_index]))

        return decoded

    @torch.no_grad()
    def infer(self, data_to_infer: Batch) -> List[torch.Tensor]:
        self.eval()

        infered = []

        for di in range(len(data_to_infer)):
            each_data = data_to_infer[di]

            encoded = self.encode(each_data)

            node_pairs = torch.combinations(torch.arange(each_data.num_nodes), r=2).to(Configuration.DEVICE)

            connection_probabilities = self.decode(encoded, node_pairs.t()).sigmoid()

            topk_indices = torch.topk(connection_probabilities, k=Configuration.TOPK_TO_INFER).indices

            connected_pairs = node_pairs[topk_indices]

            filtered_pairs = [[], []]
            for pair in connected_pairs:
                if pair.tolist() in each_data.edge_index.t().tolist():
                    continue
                elif pair.tolist()[::-1] in each_data.edge_index.t().tolist():
                    continue

                filtered_pairs[0].append(pair[0].item())
                filtered_pairs[1].append(pair[1].item())

            infered.append(torch.tensor(filtered_pairs).to(Configuration.DEVICE))

        self.train()

        return infered


class PolygonSegmenterTrainer:
    def __init__(
        self, dataset: PolygonGraphDataset, model: nn.Module, pre_trained_path: str = None, is_debug_mode: bool = False
    ):
        self.dataset = dataset
        self.model = model
        self.pre_trained_path = pre_trained_path
        self.is_debug_mode = is_debug_mode

        if self.is_debug_mode:
            add_debugvisualizer(globals())

        self._set_summary_writer()
        self._set_loss_function()
        self._set_optimizer()
        self._set_lr_scheduler()

        if self.states:
            self.model.load_state_dict(self.states["model_state_dict"])
            self.optimizer.load_state_dict(self.states["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(self.states["lr_scheduler_state_dict"])
            print(f"Set pre-trained all states from {self.states_path} \n")

    def _set_summary_writer(self):
        self.log_dir = os.path.join(Configuration.LOG_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.pre_trained_path is not None:
            self.log_dir = self.pre_trained_path

        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        self.states_dir = os.path.join(self.log_dir, "states")
        os.makedirs(self.states_dir, exist_ok=True)

        self.states = {}
        self.states_path = os.path.join(self.states_dir, Configuration.STATES_PTH)
        if len(os.listdir(self.states_dir)) > 0:
            self.states = torch.load(self.states_path)
            print(f"Load pre-trained states from {self.states_path} \n")

    def _set_loss_function(self):
        self.loss_function = nn.BCEWithLogitsLoss()

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Configuration.LEARNING_RATE)

    def _set_lr_scheduler(self):
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, factor=0.1)

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
        loss_function: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """_summary_

        Args:
            dataset (PolygonGraphDataset): _description_
            model (nn.Module): _description_
            loss_function (nn.modules.loss._Loss): _description_
            optimizer (torch.optim.Optimizer): _description_
            epoch (int): _description_

        Returns:
            float: _description_
        """

        train_losses = []
        for data_to_train in tqdm(dataset.train_dataloader, desc=f"Training... epoch: {epoch}/{Configuration.EPOCH}"):
            train_decoded = model(data_to_train)

            train_labels = self._get_labels(data_to_train)

            train_loss = loss_function(train_decoded, train_labels)

            train_losses.append(train_loss.item())

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

        return sum(train_losses) / len(train_losses)

    @torch.no_grad()
    def _evaluate_each_epoch(
        self, dataset: PolygonGraphDataset, model: nn.Module, loss_function: nn.modules.loss._Loss, epoch: int
    ):
        model.eval()

        validation_losses = []
        for data_to_validate in tqdm(
            dataset.validation_dataloder, desc=f"Evaluating... epoch: {epoch}/{Configuration.EPOCH}"
        ):
            validation_decoded = model(data_to_validate)

            validation_labels = self._get_labels(data_to_validate)

            validation_loss = loss_function(validation_decoded, validation_labels)

            validation_losses.append(validation_loss.item())

        model.train()

        return sum(validation_losses) / len(validation_losses)

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

        irregular_train_segmentation_indices = model.infer(irregular_train_batch)
        regular_train_segmentation_indices = model.infer(regular_train_batch)

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

        irregular_validation_segmentation_indices = model.infer(irregular_validation_batch)
        regular_validation_segmentation_indices = model.infer(regular_validation_batch)

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

        best_loss = torch.inf
        start = 1

        if len(self.states) > 0:
            start = self.states["epoch"] + 1
            best_loss = self.states["best_loss"]

        for epoch in range(start, Configuration.EPOCH + 1):
            avg_train_loss = self._train_each_epoch(self.dataset, self.model, self.loss_function, self.optimizer, epoch)
            avg_validation_loss = self._evaluate_each_epoch(self.dataset, self.model, self.loss_function, epoch)

            self.lr_scheduler.step(avg_validation_loss)

            if avg_validation_loss < best_loss:
                best_loss = avg_validation_loss

                states = {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "configuration": {k: v for k, v in Configuration()},
                }

                torch.save(states, self.states_path)

            else:
                states = torch.load(self.states_path)
                states.update({"epoch": epoch})

                torch.save(states, self.states_path)

            self.summary_writer.add_scalar("train_loss", avg_train_loss, epoch)
            self.summary_writer.add_scalar("validation_loss", avg_validation_loss, epoch)

            self.evaluate_qualitatively(self.dataset, self.model, epoch)

            print(f"Epoch: {epoch}th Train Loss: {avg_train_loss}, Val Loss: {avg_validation_loss}")


if __name__ == "__main__":
    Configuration.set_seed()

    dataset = PolygonGraphDataset()
    model = PolygonSegmenterGCNConv(
        in_channels=dataset.regular_polygons[0].x.shape[1],
        hidden_channels=Configuration.HIDDEN_CHANNELS,
        out_channels=Configuration.OUT_CHANNELS,
        activation_function=nn.PReLU().to(Configuration.DEVICE),
    )

    polygon_segmenter_trainer = PolygonSegmenterTrainer(
        dataset=dataset, model=model, is_debug_mode=True, pre_trained_path=None
    )
    polygon_segmenter_trainer.train()
