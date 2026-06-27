import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from matplotlib.figure import Figure

import wandb
from obf_reps.types import LoggingData


class Logger(ABC):

    @abstractmethod
    def __init__(
        self,
        log_file: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            log_file: Where to log to.
            username: User doing the logging (required if logging is personalized,
                for example when using WandB).
            metadata: Additional key value metadata that describes the run.
        """
        ...

    @abstractmethod
    def log(self, data: Dict[str, LoggingData]) -> None: ...

    @abstractmethod
    def log_to_table(self, data: List[LoggingData], table_name: str) -> None: ...

    @abstractmethod
    def create_table(self, table_name: str, columns: List[str]) -> None: ...

    @abstractmethod
    def log_tables(self) -> None: ...

    @abstractmethod
    def log_table_name(self, table_name: str) -> None: ...


class DummyLogger(Logger):

    def __init__(
        self,
        log_file: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.tables = {}
        pass

    def log(self, data: Dict[str, LoggingData]) -> None:
        pass

    def log_to_table(self, data: List[LoggingData], table_name: str) -> None:
        pass

    def create_table(self, table_name: str, columns: List[str]) -> None:
        self.tables[table_name] = None
        pass

    def log_tables(self) -> None:
        pass

    def log_table_name(self, table_name: str) -> None:
        pass


class PrintLogger(Logger):
    """Logs metrics to stdout. No wandb / account needed — use for local runs."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.tables: Dict[str, List] = {}
        self._columns: Dict[str, List[str]] = {}

    @staticmethod
    def _san(v):
        import numbers

        if isinstance(v, bool) or isinstance(v, numbers.Number) or isinstance(v, str):
            return v
        return f"<{type(v).__name__}>"

    def log(self, data: Dict[str, "LoggingData"]) -> None:
        msg = {k: self._san(v) for k, v in data.items()}
        print(f"[LOG] {msg}", flush=True)

    def log_to_table(self, data: List["LoggingData"], table_name: str) -> None:
        cols = self._columns.get(table_name)
        row = [self._san(v) for v in data]
        printable = dict(zip(cols, row)) if cols else row
        print(f"[TABLE:{table_name}] {printable}", flush=True)
        self.tables.setdefault(table_name, []).append(row)

    def create_table(self, table_name: str, columns: List[str]) -> None:
        self._columns[table_name] = columns
        self.tables[table_name] = []

    def log_tables(self) -> None:
        for name, rows in self.tables.items():
            if rows:
                print(f"\n===== TABLE {name} =====", flush=True)
                cols = self._columns.get(name)
                if cols:
                    print(cols, flush=True)
                for r in rows:
                    print(r, flush=True)

    def log_table_name(self, table_name: str) -> None:
        rows = self.tables.get(table_name, [])
        print(f"\n===== TABLE {table_name} =====", flush=True)
        cols = self._columns.get(table_name)
        if cols:
            print(cols, flush=True)
        for r in rows:
            print(r, flush=True)


class WAndBLogger(Logger):

    def __init__(
        self,
        log_file: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:

        assert log_file is not None
        assert username is not None
        assert metadata is not None

        # Check for run tags
        if "tags" in metadata:
            tags = [metadata["tag"]]
        else:
            tags = None

        self.run = wandb.init(
            project=log_file,
            entity=username,
            name=self.gen_run_name(metadata),
            config=metadata,
        )
        print(f"WANDB RUN ID: {self.run.id}")

        # table_name <> wandb.Table dictionary
        self.tables: Dict[str, wandb.Table] = {}

    def gen_run_name(self, metadata: Dict[str, str]) -> str:
        return "-".join(
            [
                metadata["obfus_data_module"],
                metadata["concept_data_module"],
                metadata["optimizer"],
                metadata["loss"],
                metadata["metric"],
                metadata["model_cls"],
            ]
        )

    def _sanatize_log_data(self, data: List[LoggingData]):

        converted_data = []
        for item in data:
            if isinstance(item, torch.Tensor):
                item = item.squeeze().cpu().numpy()
                assert len(item.shape) == 2
                converted_data.append(wandb.Image(item))
            elif isinstance(item, np.ndarray):
                assert len(item.shape) == 2
                converted_data.append(wandb.Image(item))
            elif isinstance(item, Figure):
                converted_data.append(wandb.Image(item))
            else:
                converted_data.append(item)

        return converted_data

    def log(self, data: Dict[str, LoggingData]):
        self.run.log(data)

    def log_to_table(self, data: List[LoggingData], table_name: str):

        assert table_name in self.tables
        num_columns = len(self.tables[table_name].columns)
        if len(data) != num_columns:
            raise ValueError(f"Data length {len(data)} does not match table size {num_columns}")

        converted_data = self._sanatize_log_data(data)

        self.tables[table_name].add_data(*converted_data)

    def create_table(self, table_name: str, columns: List[str]):
        self.tables[table_name] = wandb.Table(columns=columns)

    def log_tables(self) -> None:
        self.run.log(self.tables)

    def log_table_name(self, table_name: str) -> None:

        table = wandb.Table(
            columns=self.tables[table_name].columns, data=self.tables[table_name].data
        )
        self.run.log({table_name: table})

    def __del__(self):
        print(f"WANDB RUN ID: {self.run.id}")
        self.log_tables()
