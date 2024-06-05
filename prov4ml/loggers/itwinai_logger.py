
import os
import pickle
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal

class LogMixin(metaclass=ABCMeta):
    @abstractmethod
    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log ``item`` with ``identifier`` name of ``kind`` type at ``step``
        time step.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """


class Logger(LogMixin, metaclass=ABCMeta):
    """Base class for logger

    Args:
        savedir (str, optional): filesystem location where logs are stored.
            Defaults to 'mllogs'.
        log_freq (Union[int, Literal['epoch', 'batch']], optional):
            how often should the logger fulfill calls to the `log()`
            method:

            - When set to 'epoch', the logger logs only if ``batch_idx``
              is not passed to the ``log`` method.

            - When an integer
              is given, the logger logs if ``batch_idx`` is a multiple of
              ``log_freq``.

            - When set to ``'batch'``, the logger logs always.

            Defaults to 'epoch'.
    """
    #: Location on filesystem where to store data.
    savedir: str = None
    #: Supported logging 'kind's.
    supported_types: List[str]
    _log_freq: Union[int, Literal['epoch', 'batch']]

    def __init__(
        self,
        savedir: str = 'mllogs',
        log_freq: Union[int, Literal['epoch', 'batch']] = 'epoch'
    ) -> None:
        self.savedir = savedir
        self.log_freq = log_freq

    @property
    def log_freq(self) -> Union[int, Literal['epoch', 'batch']]:
        """Get ``log_feq``, namely how often should the logger
        fulfill or ignore calls to the `log()` method."""
        return self._log_freq

    @log_freq.setter
    def log_freq(self, val: Union[int, Literal['epoch', 'batch']]):
        """Sanitize log_freq value."""
        if val in ['epoch', 'batch'] or (isinstance(val, int) and val > 0):
            self._log_freq = val
        else:
            raise ValueError(
                "Wrong value for 'log_freq'. Supported values are: "
                f"['epoch', 'batch'] or int > 0. Received: {val}"
            )

    @contextmanager
    def start_logging(self):
        """Start logging context.

        Example:


        >>> with my_logger.start_logging():
        >>>     my_logger.log(123, 'value', kind='metric', step=0)


        """
        try:
            self.create_logger_context()
            yield
        finally:
            self.destroy_logger_context()

    @abstractmethod
    def create_logger_context(self):
        """Initialize logger."""

    @abstractmethod
    def destroy_logger_context(self):
        """Destroy logger."""

    @abstractmethod
    def save_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters.

        Args:
            params (Dict[str, Any]): hyperparameters dictionary.
        """

    def serialize(self, obj: Any, identifier: str) -> str:
        """Serializes object to disk and returns its path.

        Args:
            obj (Any): item to save.
            identifier (str): identifier of the item to log (expected to be a
                path under ``self.savedir``).

        Returns:
            str: local path of the serialized object to be logged.
        """
        itm_path = os.path.join(self.savedir, identifier)
        with open(itm_path, 'wb') as itm_file:
            pickle.dump(obj, itm_file)

    def should_log(
        self,
        batch_idx: Optional[int] = None
    ) -> bool:
        """Determines whether the logger should fulfill or ignore calls to the
        `log()` method, depending on the ``log_freq`` property:

        - When ``log_freq`` is set to 'epoch', the logger logs only if
          ``batch_idx`` is not passed to the ``log`` method.

        - When ``log_freq`` is an integer
          is given, the logger logs if ``batch_idx`` is a multiple of
          ``log_freq``.

        - When ``log_freq`` is set to ``'batch'``, the logger logs always.

        Args:
            batch_idx (Optional[int]): the dataloader batch idx, if available.
                Defaults to None.

        Returns:
            bool: True if the logger should log, False otherwise.
        """
        if batch_idx is not None:
            if isinstance(self.log_freq, int):
                if batch_idx % self.log_freq == 0:
                    return True
                return False
            if self.log_freq == 'batch':
                return True
            return False
        return True

