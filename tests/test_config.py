from copy import deepcopy
import pytest
from bitaboost.config import validate_config
BASE={"data":{"target_col":"control_success","validation_season":2024},"runtime":{"physical_gpu":"2","catboost_device":"0"},"catboost":{"iterations":600,"depth":8}}
def test_single_gpu_guard():
    validate_config(deepcopy(BASE)); bad=deepcopy(BASE); bad["runtime"]["catboost_device"]="0,1"
    with pytest.raises(ValueError): validate_config(bad)
