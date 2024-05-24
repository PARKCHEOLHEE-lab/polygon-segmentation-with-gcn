import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import torch
import torch.nn as nn
import datetime

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

from polygon_segmentation_with_gcn.src.commonutils import runtime_calculator
from polygon_segmentation_with_gcn.src.config import Configuration
from polygon_segmentation_with_gcn.src.dataset import PolygonGraphDataset
from polygon_segmentation_with_gcn.src.data_creator import DataCreatorHelper


class PolygonSegmenterGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.main = Sequential(
            input_args="x, edge_index, edge_weight",
            modules=[
                (GCNConv(in_channels, hidden_channels), "x, edge_index, edge_weight -> x"),
                nn.ReLU(inplace=True),
                nn.Dropout(Configuration.DROPOUT_RATE),
                (GCNConv(hidden_channels, hidden_channels), "x, edge_index, edge_weight -> x"),
                nn.ReLU(inplace=True),
                nn.Dropout(Configuration.DROPOUT_RATE),
                (GCNConv(hidden_channels, hidden_channels), "x, edge_index, edge_weight -> x"),
                nn.ReLU(inplace=True),
                nn.Dropout(Configuration.DROPOUT_RATE),
                (GCNConv(hidden_channels, hidden_channels), "x, edge_index, edge_weight -> x"),
                nn.ReLU(inplace=True),
                nn.Dropout(Configuration.DROPOUT_RATE),
                (GCNConv(hidden_channels, hidden_channels), "x, edge_index, edge_weight -> x"),
                nn.ReLU(inplace=True),
                nn.Dropout(Configuration.DROPOUT_RATE),
                (GCNConv(hidden_channels, out_channels), "x, edge_index, edge_weight -> x"),
            ],
        )

        self.to(Configuration.DEVICE)

    def encode(self, data: Batch) -> torch.Tensor:
        return self.main(data.x, data.edge_index, edge_weight=data.edge_weight)

    def decode(self, encoded: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        return (encoded[edge_label_index[0]] * encoded[edge_label_index[1]]).sum(dim=1)

    @torch.no_grad()
    def infer(self, data_to_infer: Batch) -> List[torch.Tensor]:
        self.eval()

        segmentation_indices = []

        for di in range(len(data_to_infer)):
            each_data = data_to_infer[di]

            encoded = self.encode(each_data)

            mask_to_ignore = ~torch.eye(each_data.x.shape[0], dtype=bool).to(Configuration.DEVICE)
            mask_to_ignore[each_data.edge_index[0], each_data.edge_index[1]] = False
            mask_to_ignore[each_data.edge_index[1], each_data.edge_index[0]] = False

            connection_probability = encoded @ encoded.t()
            connection_probability *= mask_to_ignore.long()

            infered = (connection_probability > Configuration.CONNECTIVITY_THRESHOLD).nonzero().t()
            segmentation_indices.append(infered)

        self.train()

        return segmentation_indices

    def forward(self, data: Batch) -> torch.Tensor:
        # Encode the features of polygon graphs
        encoded = self.encode(data)

        # Sample negative edges
        negative_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.shape[1],
            method="sparse",
        )

        # Decode the encoded features of the nodes to predict whether the edges are connected
        decoded = self.decode(encoded, edge_label_index=torch.hstack([data.edge_label_index, negative_edge_index]))

        return decoded


class PolygonSegmenterTrainer:
    def __init__(self, dataset: PolygonGraphDataset, model: nn.Module, pre_trained_path: str = None):
        self.dataset = dataset
        self.model = model
        self.pre_trained_path = pre_trained_path

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

        self.images_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

    def _set_loss_function(self):
        self.loss_function = nn.BCEWithLogitsLoss()

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Configuration.LEARNING_RATE)

    def _set_lr_scheduler(self):
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, factor=0.1)

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

            train_labels = torch.hstack([data_to_train.edge_label, torch.zeros_like(data_to_train.edge_label)])
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

            validation_labels = torch.hstack(
                [data_to_validate.edge_label, torch.zeros_like(data_to_validate.edge_label)]
            )
            validation_loss = loss_function(validation_decoded, validation_labels)

            validation_losses.append(validation_loss.item())

        model.train()

        return sum(validation_losses) / len(validation_losses)

    def evaluate(self, dataset: PolygonGraphDataset, model: nn.Module) -> None:
        """_summary_

        Args:
            dataset (PolygonGraphDataset): _description_
            model (nn.Module): _description_
        """

        c = 4

        train_indices_to_viz = torch.randperm(len(dataset.train_dataloader))[:c]
        train_subset = Subset(dataset.train_dataloader.dataset, train_indices_to_viz)
        train_sampled = DataLoader(train_subset, batch_size=c)

        validation_indices_to_viz = torch.randperm(len(dataset.validation_dataloder))[:c]
        validation_subset = Subset(dataset.validation_dataloder.dataset, validation_indices_to_viz)
        validation_sampled = DataLoader(validation_subset, batch_size=c)

        train_batch = [data_to_infer for data_to_infer in train_sampled][0]
        validation_batch = [data_to_infer for data_to_infer in validation_sampled][0]

        train_segmentation_indices = model.infer(train_batch)
        for si in range(len(train_segmentation_indices)):
            each_data = train_batch[si]
            each_segmentation_indices = train_segmentation_indices[si]

            polygon = geometry.Polygon(each_data.x[:, :2].detach().cpu().numpy())
            predicted_edges = DataCreatorHelper.connect_polygon_segments_by_indices(polygon, each_segmentation_indices)
            label_edges = DataCreatorHelper.connect_polygon_segments_by_indices(
                polygon, each_data.edge_label_index_only
            )

            print(predicted_edges)
            print(label_edges)

            # self.summary_writer.add_figure(
            #     f"train_segmentation_{si}", Plotter.plot_polygon_segmentation(polygon, predicted_edges, label_edges)
            # )

        validation_segmentation_indices = model.infer(validation_batch)
        print(validation_segmentation_indices)

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
                }

                torch.save(states, self.states_path)

                print(f"Epoch: {epoch}th Train Loss: {avg_train_loss}, Val Loss: {avg_validation_loss}")

            else:
                states = torch.load(self.states_path)
                states.update({"epoch": epoch})

                torch.save(states, self.states_path)

            self.summary_writer.add_scalar("train_loss", avg_train_loss, epoch)
            self.summary_writer.add_scalar("validation_loss", avg_validation_loss, epoch)

            self.evaluate(self.dataset, self.model)


if __name__ == "__main__":
    # from debugvisualizer.debugvisualizer import Plotter

    Configuration.set_seed()

    dataset = PolygonGraphDataset()
    model = PolygonSegmenterGCN(
        in_channels=dataset.regular_polygons[0].x.shape[1],
        hidden_channels=Configuration.HIDDEN_CHANNELS,
        out_channels=Configuration.OUT_CHANNELS,
    )

    polygon_segmenter_trainer = PolygonSegmenterTrainer(dataset=dataset, model=model)
    polygon_segmenter_trainer.train()
