import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import torch
import torch.nn as nn

from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils import negative_sampling
from polygon_segmentation_with_gcn.src.commonutils import runtime_calculator
from polygon_segmentation_with_gcn.src.config import Configuration
from polygon_segmentation_with_gcn.src.dataset import PolygonGraphDataset


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
                (GCNConv(hidden_channels, out_channels), "x, edge_index, edge_weight -> x"),
            ],
        )

        self.to(Configuration.DEVICE)

    def encode(self, data: Batch) -> torch.Tensor:
        return self.main(data.x, data.edge_index, edge_weight=data.edge_weight)

    def decode(self, encoded: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        return (encoded[edge_label_index[0]] * encoded[edge_label_index[1]]).sum(dim=1)

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
    def __init__(
        self,
        dataset: PolygonGraphDataset,
        model: nn.Module,
        loss_function: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
    ):
        self.dataset = dataset
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

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
            decoded = model(data_to_train)

            labels = torch.hstack([data_to_train.edge_label, torch.zeros_like(data_to_train.edge_label)])
            loss = loss_function(decoded, labels)

            train_losses.append(loss.item())

            model.zero_grad()
            loss.backward()
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
            decoded = model(data_to_validate)

            labels = torch.hstack([data_to_validate.edge_label, torch.zeros_like(data_to_validate.edge_label)])
            loss = loss_function(decoded, labels)

            validation_losses.append(loss.item())

        model.train()

        return sum(validation_losses) / len(validation_losses)

    def train(self):
        for epoch in range(1, Configuration.EPOCH + 1):
            avg_train_loss = self._train_each_epoch(self.dataset, self.model, self.loss_function, self.optimizer, epoch)
            avg_validation_loss = self._evaluate_each_epoch(self.dataset, self.model, self.loss_function, epoch)

            print(avg_train_loss)
            print(avg_validation_loss)

            if epoch == 2:
                break


if __name__ == "__main__":
    # from debugvisualizer.debugvisualizer import Plotter
    # from shapely import geometry

    Configuration.set_seed()

    dataset = PolygonGraphDataset()
    model = PolygonSegmenterGCN(
        in_channels=dataset.regular_polygons[0].x.shape[1],
        hidden_channels=Configuration.HIDDEN_CHANNELS,
        out_channels=Configuration.OUT_CHANNELS,
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Configuration.LEARNING_RATE)

    polygon_segmenter_trainer = PolygonSegmenterTrainer(
        dataset=dataset, model=model, loss_function=loss_function, optimizer=optimizer
    )

    polygon_segmenter_trainer.train()
