import os
from typing import List, Union

import matplotlib.container as mcontainer
import matplotlib.dates as mdates
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from marvl import processors


class WQFluxPlotter:
    def __init__(self, file_path: str, resample_rule: str = "W"):
        self.processor = processors.WQFluxProcessor(file_path)
        self.registry = {
            "resample_rules": {
                "D": {
                    "y_label": "Daily",
                    "x_locator": mdates.WeekdayLocator(),
                    "x_formatter": mdates.DateFormatter("%d-%m"),
                },
                "W": {
                    "y_label": "Weekly",
                    "x_locator": mdates.MonthLocator(),
                    "x_formatter": mdates.DateFormatter("%m-%y"),
                },
                "ME": {
                    "y_label": "Monthly",
                    "x_locator": mdates.MonthLocator(),
                    "x_formatter": mdates.DateFormatter("%m-%y"),
                },
            },
            "load": {
                "tn": {"y_label": "%FREQ% TN load (tonnes)"},
                "no3": {"y_label": "%FREQ% NO$_3$ load (tonnes)"},
                "nh4": {"y_label": "%FREQ% NH$_4$ load (tonnes)"},
                "dn": {"y_label": "%FREQ% DN load (tonnes)"},
            },
            "discharge": {"y_label": "%FREQ% mean flowrate (m$^3$/s)"},
        }
        self.resample_rule = resample_rule

    def _set_param_dict_defaults(self, param_dict: dict, defaults_dict: dict) -> None:
        for k, v in defaults_dict.items():
            if k not in param_dict:
                param_dict[k] = v

    def _update_y_label(self, label: str, match: str = "%FREQ%") -> str:
        freq = self.registry["resample_rules"][self.resample_rule]["y_label"]
        if match in label:
            label = label.replace(match, freq)
        return label

    @property
    def resample_rule(self) -> str:
        return self._resample_rule

    @resample_rule.setter
    def resample_rule(self, resample_rule: str) -> None:
        supported_rules = list(self.registry["resample_rules"].keys())
        if resample_rule not in supported_rules:
            raise ValueError(
                f"Invalid resample_rule: {resample_rule}. "
                f"Valid resample_rules are: {supported_rules}"
            )
        self._resample_rule = resample_rule

    @staticmethod
    def plot_diverging_bar_part(ax: Axes, data, param_dict: dict = {}):
        existing_bars = [
            c for c in ax.containers if isinstance(c, mcontainer.BarContainer)
        ]

        if existing_bars and "bottom" not in param_dict:
            num_bars = len(existing_bars[0])
            positive_bottoms = np.zeros(num_bars)
            negative_bottoms = np.zeros(num_bars)
            for container in existing_bars:
                for i, bar in enumerate(container):
                    height = bar.get_height()
                    if height > 0:
                        positive_bottoms[i] += height
                    else:
                        negative_bottoms[i] += height
            bottoms = np.zeros(num_bars)
            for i, value in enumerate(data.values):
                if value > 0:
                    bottoms[i] = positive_bottoms[i]
                else:
                    bottoms[i] = negative_bottoms[i]
            param_dict["bottom"] = bottoms

        out = ax.bar(data.index, data.values, **param_dict)
        param_dict.clear()
        return out

    def plot_watermark(self, fig: Figure, param_dict: dict = {}):
        file_path = self.processor.file_path
        basename = os.path.basename(file_path).split(".")[0]
        parts = os.path.dirname(file_path).split(os.sep)
        version_num = 'VX.X.X'
        if "model" in parts:
            index = parts.index("model")
            truncated_path = os.sep.join(parts[index:])
            for part in truncated_path.split(os.sep):
                if part.startswith("V"):
                    version_num = part
        watermark = f"{version_num}_{basename}"
        defaults = {
            "fontsize": 9,
            "color": "gray",
            "alpha": 0.25,
            "ha": "center",
            "va": "center",
            "rotation": 0,
        }
        self._set_param_dict_defaults(param_dict, defaults)
        out = fig.text(0.5, 0.075, watermark, **param_dict)
        param_dict.clear()
        return out

    def plot_split_fill(
        self,
        ax: Axes,
        top_label: str,
        bottom_label: str,
        split_at_y: float = 0,
        top_param_dict: dict = {},
        bottom_param_dict: dict = {},
    ):
        default_top_params = {
            "color": "red",
            "label": top_label,
            "zorder": 0,
            "alpha": 0.075,
        }
        default_bottom_params = {
            "color": "blue",
            "label": bottom_label,
            "zorder": 0,
            "alpha": 0.075,
        }
        self._set_param_dict_defaults(top_param_dict, default_top_params)
        self._set_param_dict_defaults(bottom_param_dict, default_bottom_params)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.autoscale(False)

        out_1 = ax.fill_between(xlim, split_at_y, ylim[1], **top_param_dict)
        out_2 = ax.fill_between(xlim, ylim[0], split_at_y, **bottom_param_dict)
        top_param_dict.clear()
        bottom_param_dict.clear()
        return [out_1, out_2]

    def plot_load(
        self,
        ax: Axes,
        load_type: str,
        nodestring_id: int,
        invert: bool = False,
        param_dict: dict = {},
    ):
        load = self.processor.get_load(
            load_type, nodestring_id, invert, self.resample_rule
        )
        defaults = {
            "width": 4,
            "alpha": 1,
            "edgecolor": "black",
            "linewidth": 1.0,
            "color": "black",
            "label": f"NS{nodestring_id}",
        }
        self._set_param_dict_defaults(param_dict, defaults)
        out = self.plot_diverging_bar_part(ax, load, param_dict)
        self.style_axes(ax, y_label=self.registry["load"][load_type]["y_label"])
        param_dict.clear()
        return out

    def plot_net_load(
        self,
        ax: Axes,
        load_type: str,
        nodestring_ids: List[int],
        invert: List[bool],
        param_dict: dict = {},
    ):
        load = self.processor.get_net_load(
            load_type, nodestring_ids, invert, self.resample_rule
        )
        defaults = {
            "color": "mediumseagreen",
            "linestyle": "-",
            "linewidth": 1.5,
            "label": "Net import",
        }
        self._set_param_dict_defaults(param_dict, defaults)
        out = ax.plot(load.index, load.values, **param_dict)
        self.style_axes(ax, y_label=self.registry["load"][load_type]["y_label"])
        param_dict.clear()
        return out

    def plot_discharge(self, ax: Axes, nodestring_id: int, param_dict: dict = {}):
        discharge = self.processor.get_discharge(nodestring_id, self.resample_rule)
        defaults = {"color": "black", "linewidth": 1.5}
        self._set_param_dict_defaults(param_dict, defaults)
        out = ax.plot(discharge.index, discharge.values, **param_dict)
        self.style_axes(ax, y_label=self.registry["discharge"]["y_label"])
        param_dict.clear()
        return out

    def plot_mean_discharge(
        self, ax: Axes, nodestring_id: int, num_std: int = 0, param_dict: dict = {}
    ):
        discharge = self.processor.get_discharge(nodestring_id, self.resample_rule)
        mean = discharge.mean()
        std = discharge.std()
        if num_std != 0:
            mean = mean + num_std * std
            if num_std > 0:
                label = f"Mean + {num_std}σ: {mean:.0f} m$^3$/s"
            else:
                label = f"Mean - {abs(num_std)}σ: {mean:.0f} m$^3$/s"
        else:
            label = f"Mean: {mean:.0f} m$^3$/s"
        defaults = {"color": "black", "linestyle": "--", "alpha": 0.5, "label": label}
        self._set_param_dict_defaults(param_dict, defaults)
        out = ax.axhline(y=mean, **param_dict)
        param_dict.clear()
        return out

    def style_axes(self, ax: Axes, y_label: Union[str, None] = None):
        if y_label is not None:
            y_label = self._update_y_label(y_label)
            ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        locator = self.registry["resample_rules"][self.resample_rule]["x_locator"]
        formatter = self.registry["resample_rules"][self.resample_rule]["x_formatter"]
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
