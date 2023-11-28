import pytest

import pastas as ps
from pastas.plugins.plugins import PluginRegistrationError

from . import fake_plugin as my_plugin


def test_register_plugin_in_root():
    # register in root
    ps.plugins.register(my_plugin, target=None)
    assert hasattr(ps, "FakeStressModel")


def test_register_plugin_in_existing_submodule():
    # register in existing ps.stressmodels
    ps.plugins.register(my_plugin, target="stressmodels")
    assert hasattr(ps.stressmodels, "FakeStressModel")


def test_register_plugin_in_nonexisting_submodule():
    # register in non-existing sub module
    ps.plugins.register(my_plugin, target="plugins")
    assert hasattr(ps.plugins, "FakeStressModel")


def test_register_plugin_with_existing_name():
    # plugin already registered in second test
    # registering again should raise error
    with pytest.raises(PluginRegistrationError) as excinfo:
        ps.plugins.register(my_plugin, target="plugins")

    assert str(excinfo.value) == (
        "Failed registering plugin, <pastas.plugins> already "
        "contains class or method named 'FakeStressModel'."
    )
