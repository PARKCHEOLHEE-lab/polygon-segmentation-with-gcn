import os
import sys

if os.path.abspath(os.path.join(__file__, "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

if os.path.abspath(os.path.join(__file__, "../../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))

import torch

from typing import Tuple, List
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.data import DataLoader, Dataset, Data
from polygon_segmentation_with_gcn.src.config import Configuration


class RegularPolygonDataset(Dataset):
    def __init__(self, data_path: str = Configuration.MERGED_SAVE_PATH):
        self.regular_polygons: List[Data]
        self.regular_polygons = []
        for file in os.listdir(data_path):
            if Configuration.LANDS_DATA_REGULAR_PT in file:
                self.regular_polygons.extend(torch.load(os.path.join(data_path, file)))

    def __len__(self):
        return len(self.regular_polygons)

    def __getitem__(self, index):
        return self.regular_polygons[index]


class IrregularPolygonDataset(Dataset):
    def __init__(self, data_path: str = Configuration.MERGED_SAVE_PATH):
        self.irregular_polygons: List[Data]
        self.irregular_polygons = []
        for file in os.listdir(data_path):
            if Configuration.LANDS_DATA_IRREGULAR_PT in file:
                self.irregular_polygons.extend(torch.load(os.path.join(data_path, file)))

    def __len__(self):
        return len(self.irregular_polygons)

    def __getitem__(self, index):
        return self.irregular_polygons[index]


class PolygonGraphDataset(Dataset):
    def __init__(self, regular_polygons: RegularPolygonDataset, irregular_polygons: IrregularPolygonDataset):
        self.regular_polygons = regular_polygons
        self.irregular_polygons = irregular_polygons

        self.train_dataloader, self.validation_dataloder, self.test_dataloader = self._get_dataloaders(
            self.regular_polygons, self.irregular_polygons
        )

    def _get_dataloaders(
        self, regular_polygons: RegularPolygonDataset, irregular_polygons: IrregularPolygonDataset
    ) -> Tuple[DataLoader]:
        regular_train, regular_validation, regular_test = random_split(regular_polygons, Configuration.SPLIT_RATIOS)
        irregular_train, irregular_validation, irregular_test = random_split(
            irregular_polygons, Configuration.SPLIT_RATIOS
        )

        splitted_regular = torch.tensor([len(regular_train), len(regular_validation), len(regular_test)])
        expected_regular = torch.tensor([len(regular_polygons) * ratio for ratio in Configuration.SPLIT_RATIOS])
        assert torch.all(splitted_regular == expected_regular)

        splitted_irregular = torch.tensor([len(irregular_train), len(irregular_validation), len(irregular_test)])
        expected_irregular = torch.tensor([len(irregular_polygons) * ratio for ratio in Configuration.SPLIT_RATIOS])
        assert torch.all(splitted_irregular == expected_irregular)

        train_dataset = ConcatDataset([regular_train, irregular_train])
        validation_dataset = ConcatDataset([regular_validation, irregular_validation])
        test_dataset = ConcatDataset([regular_test, irregular_test])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=Configuration.BATCH_SIZE,
            shuffle=True,
        )

        return train_dataloader, validation_dataloader, test_dataloader
