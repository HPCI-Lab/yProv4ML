
from typing import Any
from collections import namedtuple

class Prov4MLLOD:
    """
    Provides methods for creating level and value attributes for provenance logging.

    Attributes:
        LVL_1 (str): Level 1 indicator.
        LVL_2 (str): Level 2 indicator.
        lv_attr (namedtuple): Named tuple for level and value.
        lv_epoch_value (namedtuple): Named tuple for level, epoch, and value.
    """
    LVL_1 = "1"
    LVL_2 = "2"

    lv_attr = namedtuple('lv_attr', ['level', 'value'])
    lv_epoch_value = namedtuple('lv_step_attr', ['level', 'epoch', 'value'])

    @staticmethod
    def get_lv1_attr(value: Any) -> str:
        """
        Gets a level 1 attribute string representation.

        Parameters:
            value (Any): The value to associate with level 1.

        Returns:
            str: String representation of the level 1 attribute.
        """
        return str(Prov4MLLOD.lv_attr(Prov4MLLOD.LVL_1, value))

    @staticmethod
    def get_lv2_attr(value: Any) -> str:
        """
        Gets a level 2 attribute string representation.

        Parameters:
            value (Any): The value to associate with level 2.

        Returns:
            str: String representation of the level 2 attribute.
        """
        return str(Prov4MLLOD.lv_attr(Prov4MLLOD.LVL_2, value))
    
    @staticmethod
    def get_lv1_epoch_value(epoch: int, value: Any) -> str:
        """
        Gets a level 1 epoch value string representation.

        Parameters:
            epoch (int): The epoch number.
            value (Any): The value to associate with the epoch at level 1.

        Returns:
            str: String representation of the level 1 epoch value.
        """
        return str(Prov4MLLOD.lv_epoch_value(Prov4MLLOD.LVL_1, epoch, value))
    
    @staticmethod
    def get_lv2_epoch_value(epoch: int, value: Any) -> str:
        """
        Gets a level 2 epoch value string representation.

        Parameters:
            epoch (int): The epoch number.
            value (Any): The value to associate with the epoch at level 2.

        Returns:
            str: String representation of the level 2 epoch value.
        """
        return str(Prov4MLLOD.lv_epoch_value(Prov4MLLOD.LVL_2, epoch, value))
