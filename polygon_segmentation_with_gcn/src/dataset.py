import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import torch
import numpy as np

from typing import Tuple, List
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from polygon_segmentation_with_gcn.src.config import Configuration
from polygon_segmentation_with_gcn.src.commonutils import runtime_calculator


class DatasetHelper:
    def __init__(self):
        pass

    def _tidy_all_data(self, data: List[Data]):
        edi = 0
        while edi < len(data):
            each_data = data[edi]
            each_data.x = each_data.x[:, :-1]
            each_data.x = torch.tensor(each_data.x, dtype=torch.float32).to(Configuration.DEVICE)
            each_data.edge_weight = torch.tensor(each_data.edge_weight, dtype=torch.float32).to(Configuration.DEVICE)
            each_data.edge_index = torch.tensor(each_data.edge_index).to(Configuration.DEVICE)
            each_data.edge_label_index = torch.tensor(each_data.edge_label_index).to(Configuration.DEVICE)
            each_data.edge_label = torch.ones([each_data.edge_label_index.shape[1]]).to(Configuration.DEVICE)

            if all(each_data.x[:, :2][-1] == each_data.x[:, :2][-2]):
                del data[edi]
                continue

            # Dummy label
            if None in each_data.edge_label_index_only.flatten():
                edge_label_index_only = np.array([[0, 1], [1, 2]])
                edge_label_index_only = np.repeat(
                    edge_label_index_only, each_data.num_nodes * Configuration.POSITIVE_SAMPLE_MULTIPLIER, axis=1
                )

            else:
                edge_label_index_only = each_data.edge_label_index_only
                if edge_label_index_only.shape[1] == 1:
                    # edge_label_index_only = np.hstack([edge_label_index_only, np.array([[0], [1]])])
                    edge_label_index_only = np.hstack(
                        [
                            edge_label_index_only,
                            np.array([[edge_label_index_only[0][0]], [edge_label_index_only[1][0]]]),
                        ]
                    )

                edge_label_index_only = np.repeat(
                    edge_label_index_only, each_data.num_nodes * Configuration.POSITIVE_SAMPLE_MULTIPLIER, axis=1
                )

            assert edge_label_index_only.shape == (
                2,
                each_data.num_nodes * Configuration.POSITIVE_SAMPLE_MULTIPLIER * 2,
            )

            each_data.edge_label_index_only = torch.tensor(edge_label_index_only).to(Configuration.DEVICE)

            edi += 1


class RegularPolygonDataset(Dataset, DatasetHelper):
    @runtime_calculator
    def __init__(self, data_path: str = Configuration.MERGED_SAVE_PATH):
        self.regular_polygons: List[Data]
        self.regular_polygons = []
        for file in os.listdir(data_path):
            if Configuration.LANDS_DATA_REGULAR_PT in file:
                data = torch.load(os.path.join(data_path, file))
                self._tidy_all_data(data)

                self.regular_polygons.extend(data)

    def __len__(self):
        return len(self.regular_polygons)

    def __getitem__(self, index):
        return self.regular_polygons[index]


class IrregularPolygonDataset(Dataset, DatasetHelper):
    @runtime_calculator
    def __init__(self, data_path: str = Configuration.MERGED_SAVE_PATH):
        self.irregular_polygons: List[Data]
        self.irregular_polygons = []
        for file in os.listdir(data_path):
            if Configuration.LANDS_DATA_IRREGULAR_PT in file:
                data = torch.load(os.path.join(data_path, file))
                self._tidy_all_data(data)

                self.irregular_polygons.extend(data)

    def __len__(self):
        return len(self.irregular_polygons)

    def __getitem__(self, index):
        return self.irregular_polygons[index]


class PolygonGraphDataset(Dataset):
    def __init__(self, slicer: int = None):
        self.regular_polygons = RegularPolygonDataset()
        self.irregular_polygons = IrregularPolygonDataset()

        (
            self.train_dataloader,
            self.validation_dataloder,
            self.test_dataloader,
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
            self.regular_train,
            self.irregular_train,
            self.regular_validation,
            self.irregular_validation,
            self.regular_test,
            self.irregular_test,
        ) = self._get_dataloaders(self.regular_polygons, self.irregular_polygons, slicer)

    def _get_dataloaders(
        self, regular_polygons: RegularPolygonDataset, irregular_polygons: IrregularPolygonDataset, slicer: int
    ) -> Tuple[DataLoader]:
        regular_train, regular_validation, regular_test = random_split(regular_polygons, Configuration.SPLIT_RATIOS)
        irregular_train, irregular_validation, irregular_test = random_split(
            irregular_polygons, Configuration.SPLIT_RATIOS
        )

        train_dataset = ConcatDataset([regular_train, irregular_train])
        validation_dataset = ConcatDataset([regular_validation, irregular_validation])
        test_dataset = ConcatDataset([regular_test, irregular_test])

        if slicer is not None:
            train_dataset = ConcatDataset([regular_train.dataset[:slicer], irregular_train.dataset[:slicer]])
            validation_dataset = ConcatDataset(
                [regular_validation.dataset[:slicer], irregular_validation.dataset[:slicer]]
            )
            test_dataset = ConcatDataset([regular_test.dataset[:slicer], irregular_test.dataset[:slicer]])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        return (
            train_dataloader,
            validation_dataloader,
            test_dataloader,
            train_dataset,
            validation_dataset,
            test_dataset,
            regular_train,
            irregular_train,
            regular_validation,
            irregular_validation,
            regular_test,
            irregular_test,
        )
