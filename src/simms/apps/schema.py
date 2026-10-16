"""The native CLI schema, derived from the application functions' typed signatures."""

import inspect
import types
from typing import Annotated, Literal, Union, get_args, get_origin, get_type_hints

import click
from pydantic import ConfigDict, create_model


def inputs_from_signature(func):
    """Keep defaults, validators and descriptions in the function signature."""
    hints = get_type_hints(func, include_extras=True)
    fields = {name: (hints[name], param.default) for name, param in inspect.signature(func).parameters.items()}
    return create_model(f"{func.__name__}Inputs", __config__=ConfigDict(extra="forbid"), **fields)


def _leaves(annotation):
    origin = get_origin(annotation)
    if origin is Annotated:
        return _leaves(get_args(annotation)[0])
    if origin in (Union, types.UnionType, list, tuple):
        return [leaf for arg in get_args(annotation) if arg is not type(None) for leaf in _leaves(arg)]
    return [annotation]


def _is_list(annotation):
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return all(_is_list(arg) for arg in get_args(annotation) if arg is not type(None))
    return origin in (list, tuple)


def build_options(model):
    """Render simms' flat schemas as Click options, including their short aliases."""
    options = []
    for name, field in model.model_fields.items():
        default = None if field.is_required() else field.default
        leaves = _leaves(field.annotation)
        multiple = _is_list(field.annotation)
        flag = "--" + name.replace("_", "-")
        kwargs = dict(required=field.is_required(), help=field.description)
        if bool in leaves and not multiple:
            flag += "/--no-" + name.replace("_", "-")
            kwargs.update(is_flag=True, default=bool(default))
        else:
            choices = next((get_args(leaf) for leaf in leaves if get_origin(leaf) is Literal), None)
            scalars = set(leaves) & {str, int, float, bool}
            scalar = next(iter(scalars)) if len(scalars) == 1 else str
            kwargs.update(type=click.Choice(choices) if choices else scalar, multiple=multiple)
            if default is not None:
                kwargs["default"] = default
        metadata = (field.json_schema_extra or {}).get("param_meta", {})
        abbrev = metadata.get("abbreviation")
        options.append(click.Option([flag, *(["-" + abbrev] if abbrev else [])], **kwargs))
    return options


def optional_step(func, *, name, info):
    """Load the optional workflow adapter only when its public export is requested."""
    try:
        from shinobi import pystep
    except ModuleNotFoundError as exc:
        if exc.name != "shinobi":
            raise
        raise ImportError(
            "The workflow adapter requires simms[ninja]; install it with uv pip install 'simms[ninja]'."
        ) from exc
    return pystep(name=name, info=info)(func)
