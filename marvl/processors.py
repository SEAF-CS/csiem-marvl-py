from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd
import xarray as xr


class BaseProcessor(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str) -> None:
        self._file_path = file_path
        self.data = self._read_file(self._file_path)

    @abstractmethod
    def _read_file(self, file_path: str):
        pass


class WQFluxProcessor(BaseProcessor):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._conv = 1800 * 14.0 / (1000 * 1000000)
        self.registry = {
            'tn': {
                'wq_nit_nit': {'multiplier': None},
                'wq_nit_amm': {'multiplier': None},
                'wq_ogm_don': {'multiplier': None},
                'wq_ogm_pon': {'multiplier': None},
                'wq_phy_mixed': {'multiplier': 0.151},
                'wq_phy_pico': {'multiplier': 0.100},
                'wq_phy_diatom': {'multiplier': 0.137},
                'wq_phy_dino': {'multiplier': 0.151},
            },
            'no3': {'wq_nit_nit': {'multiplier': None}},
            'nh4': {'wq_nit_amm': {'multiplier': None}},
            'dn': {
                'wq_nit_nit': {'multiplier': None},
                'wq_nit_amm': {'multiplier': None},
            },
        }

    def _read_file(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        current_names = data.columns.tolist()
        tidy_names = []
        for name in current_names:
            if ' ' in name:
                name = name.split(' ')[0]
            tidy_name = name.lower()
            if tidy_name.startswith('ns'):
                tidy_name = tidy_name.replace('ns', '')
            tidy_names.append(tidy_name)
        data.rename(columns=dict(zip(current_names, tidy_names)), inplace=True)
        data['time'] = pd.to_datetime(data['time'], dayfirst=True)
        return data

    def _get_col_names(self, time: bool) -> List[str]:
        col_names = self.data.columns.tolist()
        if not time:
            if 'time' in col_names:
                col_names.remove('time')
        return col_names

    def _validate(
        self,
        nodestring_id: Union[int, None] = None,
        load_type: Union[str, None] = None,
    ):
        if (
            nodestring_id is not None
            and nodestring_id not in self.get_nodestring_ids()
        ):
            raise ValueError(
                f'Invalid nodestring_id: {nodestring_id}. '
                f'Valid nodestring_ids are: {self.get_nodestring_ids()}'
            )
        if load_type is not None and load_type not in self.get_load_types():
            raise ValueError(
                f'Invalid load_type: {load_type}. '
                f'Valid load_types are: {self.get_load_types()}'
            )

    def get_start_date(self) -> pd.Timestamp:
        return self.data['time'].min()

    def get_load_types(self) -> List[str]:
        return list(self.registry.keys())

    def get_nodestring_ids(self) -> List[int]:
        col_names = self._get_col_names(False)
        ns_ids = []
        for col_name in col_names:
            ns_id = int(col_name.split('_')[0])
            ns_ids.append(ns_id)
        return list(set(ns_ids))

    def get_nodestring_var_names(self, nodestring_id: int) -> List[str]:
        self._validate(nodestring_id)
        col_names = self._get_col_names(False)
        var_names = []
        for col_name in col_names:
            if col_name.startswith(str(nodestring_id) + '_'):
                var_names.append(col_name)
        return var_names

    def get_nodestring_cols(self, nodestring_id: int) -> pd.DataFrame:
        self._validate(nodestring_id)
        ns_pd = self.data.copy()
        for col in ns_pd.columns:
            if not col.startswith(str(nodestring_id)):
                ns_pd.drop(col, axis=1, inplace=True)
        return ns_pd

    def get_load(
        self,
        load_type: str,
        nodestring_id: int,
        invert: bool = False,
        resample_rule: str = 'W',
    ) -> pd.Series:
        self._validate(nodestring_id, load_type)
        self.data.set_index('time', inplace=True)
        total_load = None
        loads = []
        for k, v in self.registry[load_type].items():
            load = (
                self.data[f'{nodestring_id}_{k}'].resample(resample_rule).sum()
            )
            if v['multiplier'] is not None:
                load = load * v['multiplier']
            loads.append(load)
        total_load = sum(loads)
        total_load = total_load * self._conv
        self.data.reset_index(inplace=True)
        if invert:
            total_load = total_load * -1
        return total_load

    def get_net_load(
        self,
        load_type: Union[str, List[str]],
        nodestring_id: Union[int, List[int]],
        invert: Union[bool, List[bool]],
        resample_rule: str = 'W',
    ) -> pd.Series:
        load_type = (
            [load_type] if not isinstance(load_type, list) else load_type
        )
        nodestring_id = (
            [nodestring_id]
            if not isinstance(nodestring_id, list)
            else nodestring_id
        )
        invert = [invert] if not isinstance(invert, list) else invert
        assert len(nodestring_id) == len(invert), (
            'nodestring_id and invert must have equal lengths'
        )
        loads = []
        for i in range(len(load_type)):
            for j in range(len(nodestring_id)):
                self._validate(nodestring_id[j], load_type[i])
                load = self.get_load(
                    load_type[i], nodestring_id[j], invert[j], resample_rule
                )
                loads.append(load)
        net_load = sum(loads)
        return net_load

    def get_discharge(self, nodestring_id: int, resample_rule: str = 'W') -> pd.Series:
        self._validate(nodestring_id)
        self.data.set_index('time', inplace=True)
        discharge = (
            self.data[f'{nodestring_id}_flow'].resample(resample_rule).mean()
        )
        self.data.reset_index(inplace=True)
        return discharge


class NCProcessor(BaseProcessor):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_path = file_path

    def _read_file(self, file_path: str) -> xr.Dataset:
        data = xr.open_dataset(file_path)
        return data
