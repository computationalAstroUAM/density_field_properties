"""Registry mapping Haloscope INPUT feature names to attachers."""

from typing import Sequence, Type

from density_field_properties.preprocessing.input_features.base import InputFeatureAttacher
from density_field_properties.preprocessing.input_features.n_halos_env import (
    ENV_FEATURE_NAME,
    NHalosEnvironmentFeature,
)
from density_field_properties.preprocessing.input_features.rockstar_t_over_u import (
    T_OVER_U_FEATURE_NAME,
    RockstarTOverUFeature,
)
from density_field_properties.preprocessing.input_features.tidal_anisotropy import (
    HALO_ENVIRONMENT_DESCRIPTOR_COLUMNS,
    HaloEnvironmentDescriptorsAttacher,
)

ATTACHER_REGISTRY: dict[str, Type[InputFeatureAttacher]] = {
    ENV_FEATURE_NAME: NHalosEnvironmentFeature,
    T_OVER_U_FEATURE_NAME: RockstarTOverUFeature,
}


def _known_input_feature_names() -> list[str]:
    """
    List all supported Haloscope INPUT feature names.

    Returns
    -------
    list[str]
        Sorted unique feature keys.
    """
    names = set(ATTACHER_REGISTRY)
    names.update(HALO_ENVIRONMENT_DESCRIPTOR_COLUMNS)
    return sorted(names)


def _attacher_class(feature_name: str) -> Type[InputFeatureAttacher]:
    """
    Resolve the attacher class for one INPUT feature name.

    Parameters
    ----------
    feature_name : str
        Haloscope INPUT feature name.

    Returns
    -------
    Type[InputFeatureAttacher]
        Concrete attacher class.

    Raises
    ------
    ValueError
        If ``feature_name`` is not registered.
    """
    attacher_cls = ATTACHER_REGISTRY.get(feature_name)
    if attacher_cls is not None:
        return attacher_cls

    known = _known_input_feature_names()
    raise ValueError(f"Unknown input feature {feature_name!r}; known: {', '.join(known)}")


def _ordered_halo_environment_descriptor_columns(
    input_features: Sequence[str],
) -> tuple[str, ...]:
    """
    Collect descriptor column names from ``input_features`` in request order.

    Parameters
    ----------
    input_features : Sequence[str]
        Haloscope INPUT feature names from run configuration.

    Returns
    -------
    tuple[str, ...]
        Unique ``tidal_anisotropy`` / ``overdensity`` entries in list order.
    """
    descriptor_set = set(HALO_ENVIRONMENT_DESCRIPTOR_COLUMNS)
    return tuple(name for name in input_features if name in descriptor_set)


def resolve_attachers(input_features: Sequence[str]) -> list[InputFeatureAttacher]:
    """
    Build attacher instances for the requested Haloscope INPUT features.

    Parameters
    ----------
    input_features : Sequence[str]
        Ordered Haloscope INPUT column names.

    Returns
    -------
    list[InputFeatureAttacher]
        Attacher instances following ``input_features`` order.

    Raises
    ------
    ValueError
        If ``input_features`` is empty or contains unknown names.
    """
    if len(input_features) == 0:
        raise ValueError("input_features must contain at least one feature name")

    known = set(_known_input_feature_names())
    descriptor_columns = _ordered_halo_environment_descriptor_columns(input_features)

    attachers: list[InputFeatureAttacher] = []
    seen: set[str] = set()
    descriptor_attacher_added = False
    for feature_name in input_features:
        if feature_name in seen:
            raise ValueError(f"Duplicate input feature: {feature_name}")
        seen.add(feature_name)
        if feature_name not in known:
            raise ValueError(
                f"Unknown input feature {feature_name!r}; known: {', '.join(_known_input_feature_names())}"
            )
        if feature_name in HALO_ENVIRONMENT_DESCRIPTOR_COLUMNS:
            if not descriptor_attacher_added:
                attachers.append(HaloEnvironmentDescriptorsAttacher(descriptor_columns))
                descriptor_attacher_added = True
            continue
        attachers.append(_attacher_class(feature_name)())

    return attachers
