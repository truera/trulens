import collections

from trulens.nn.backend import get_backend
from trulens.utils import tru_logger
from trulens.utils.typing import many_of_om
from trulens.utils.typing import om_of_many


def hash_tensor(tensor):
    if hasattr(tensor, 'ref'):
        return tensor.ref()
    else:
        return tensor


def unhash_tensor(tensor_ref):
    if hasattr(tensor_ref, 'deref'):
        return tensor_ref.deref()
    else:
        return tensor_ref


def get_layer_input_paths(model):
    '''
    get_layer_input_paths Gets the nesting path to each layer's input tensor

    Parameters
    ----------
    model : keras.models.Model
        The Keras model

    Returns
    -------
    Dict[str, (str, List[str], Dict[str, str])]
        Mapping from each layer name to their input tensor's mapping from the previous layer
    '''
    B = get_backend()

    def recurse_outputs(obj, prefix=None):
        """Given a possibly nested data structure, return a mapping from Tensors to their path in the data structure.

        Primitives used:
            str: represent dictionary keys
            int: represent list indexes or dictionary keys
        """
        ret = {}
        prefix = [] if prefix is None else prefix

        if B.is_tensor(obj):
            return {hash_tensor(obj): prefix}
        elif isinstance(obj, collections.abc.Mapping):
            for key, val in obj.items():
                ret.update(recurse_outputs(val, prefix + [key]))
        elif isinstance(obj, collections.abc.Iterable):
            for i, elem in enumerate(obj):
                ret.update(recurse_outputs(elem, prefix + [i]))
        else:
            tru_logger.warning(type(obj))
        return ret

    def get_arg_path(layers, obj):
        """Given the tensor argument path, retrieve the actual tensor(s) in the matching data structure"""

        def path_from_tensor(a):
            for layer in layers:
                try:
                    return layer_outputs[layer.name][hash_tensor(a)]
                except KeyError:
                    continue
            raise ValueError(
                f'Unable to find matching layer in {layers}for arg {a}'
            )

        if B.is_tensor(obj):
            return path_from_tensor(obj)
        elif isinstance(obj, collections.abc.Mapping):
            return {key: path_from_tensor(a) for key, a in obj.items()}
        elif isinstance(obj, collections.abc.Iterable):
            return [path_from_tensor(a) for a in obj]
        else:
            raise ValueError('Invalid argument found')

    layer_outputs = {}
    layer_input_paths = {}

    # Outputs of each layer
    node_depths = sorted(list(model._nodes_by_depth.keys()), reverse=True)
    for depth in node_depths:
        if depth < 0:
            break
        for node in model._nodes_by_depth[depth]:
            # add layer output to layer_outputs
            layer = node.outbound_layer
            layer_outputs[
                layer.name
            ] = recurse_outputs(om_of_many(node.output_tensors), [layer.name])
            if node.inbound_layers:
                # Get input tensor paths for next layer from from prev layer's outputs
                if isinstance(node.inbound_layers, dict):
                    prev_layers = set(node.inbound_layers.values())
                else:
                    prev_layers = set(many_of_om(node.inbound_layers))

                try:
                    args = many_of_om(node.call_args)
                    kwargs = node.call_kwargs
                except AttributeError:
                    # call_args, call_kwargs attributes don't exist in older Keras versions
                    args = many_of_om(node.input_tensors)
                    kwargs = {}

                arg_paths = [get_arg_path(prev_layers, arg) for arg in args]
                kwarg_paths = {
                    key: get_arg_path(prev_layers, arg)
                    for key, arg in kwargs.items()
                }

                if layer.name not in layer_input_paths:
                    layer_input_paths[layer.name] = {
                        'args': arg_paths,
                        'kwargs': kwarg_paths
                    }
                else:
                    layer_input_paths[layer.name]['args'].extend(arg_paths)
                    layer_input_paths[layer.name]['kwargs'].update(kwarg_paths)

    return layer_input_paths


def get_layer_input_tensors(
    layer, layer_outputs, layer_input_paths, keras_module
):
    '''
    get_layer_input_tensors Get the args and kwargs for the layer

    Parameters
    ----------
    layer : keras.layers.Layer
        The layer to get the input tensors for

    layer_outputs : Dict[keras.layers.Layer, Tensor]
        Mapping of previous layer to layer outputs

    layer_input_paths : Dict[str, (str, List[str], Dict[str, str])]
        Mapping from each layer name to their input tensor's mapping from the previous layer

    keras_module : Python Module
        Either the keras module or tf.keras module

    Returns
    -------
    Tensors for layer from layer_outputs that match the Tensor input paths described in layer_input_paths
    args List[Tensor]
    kwargs Dict[str, Tensor]
    '''

    def lookup_arg_path(path):
        out = layer_outputs[path[0]]
        for part in path[1:]:
            if isinstance(out, collections.abc.Mapping):
                out = out[part]
            elif isinstance(part, int) and isinstance(out,
                                                      collections.abc.Iterable):
                out = out[part]
            else:
                raise ValueError(f'Invalid part path {part} for object {out}')
        return out

    def get_tensor_from_arg_path(arg):
        if isinstance(arg, collections.abc.Mapping):
            return {k: lookup_arg_path(path) for k, path in arg.items()}
        elif isinstance(arg, collections.abc.Iterable) and len(arg) >= 0:
            if isinstance(arg[0], str):
                # one path
                return lookup_arg_path(arg)
            elif isinstance(arg[0], collections.abc.Iterable):
                return [lookup_arg_path(path) for path in arg]
        else:
            raise ValueError(f'Invalid arg path {arg}')

    # No layers preceding Input layers. Use Input tensor
    if isinstance(layer, keras_module.layers.InputLayer):
        return many_of_om(layer.input), {}

    layer_name = layer.name
    input_paths = layer_input_paths[layer_name]
    args = input_paths['args']
    kwargs = input_paths['kwargs']

    # fetch input args for this layer
    input_args = [get_tensor_from_arg_path(arg) for arg in args]
    input_kwargs = {
        key: get_tensor_from_arg_path(arg) for key, arg in kwargs.items()
    }

    return input_args, input_kwargs


def rename_nested_layers(model, keras_module):
    '''
    Recursively renames layers to the full heirarchical name.
    Useful when flattening nested models but want to ensure unique layer names.

    Parameters
    ----------
    model : keras.models.Model
        Base model

    keras_module : Python Module
        Either the keras module or tf.keras module
    '''
    for layer in model.layers:
        # InputLayer names are used as keys when dictionary is passed to model.
        # Changing them may cause the model to output different results.

        # Due to the recursive flattening operation, this method can be called
        # on the same model multiple times. Skip if renaming already occurred
        if not isinstance(layer, keras_module.layers.InputLayer
                         ) and not layer.name.startswith(f'{model.name}/'):
            layer._name = f'{model.name}/{layer.name}'
            if isinstance(layer, keras_module.models.Model):
                rename_nested_layers(layer, keras_module)


def perform_replacements(model, replacements, keras_module):
    '''
    perform_replacements Clone model but replace layers in replacements

    Parameters
    ----------
    model : keras.models.Model
        Base model

    replacements : Dict[keras.layers.Layer, keras.layers.Layer]
        Mapping of keras layers with their replacement

    keras_module : Python Module
        Either the keras module or tf.keras module

    Returns
    -------
    A new keras Model that replicates model that but with layer replacements.

    '''
    layer_input_paths = get_layer_input_paths(model)
    layer_outputs = {}

    def prop_through_layer(depth, dirty=False):
        """Recursively go through the layer via model graph nodes and update layers with their replacements if necessary.

        Args:
            depth (int): The model is represented as a DAG. depth represents the node depth from the end of the DAG.
            dirty (bool, optional): True if nodes at a previous depth have been replaced. Defaults to False.

        Returns:
            List[Tensor]: The output tensors of the updated computational graph.
        """
        if depth < 0:
            nodes = model._nodes_by_depth[0]
            return om_of_many(
                [layer_outputs[node.outbound_layer.name] for node in nodes]
            )

        nodes = model._nodes_by_depth[depth]
        if not dirty and all(
                node.outbound_layer not in replacements and
                not isinstance(node.outbound_layer, keras_module.models.Model)
                for node in nodes):
            # no prior modifications, no nested models, and no replacements at this depth, continue on
            for node in nodes:
                layer_outputs[node.outbound_layer.name
                             ] = om_of_many(node.output_tensors)

            return prop_through_layer(depth=depth - 1, dirty=dirty)

        else:
            # is dirty or needs to perform replacement
            for node in nodes:
                layer = node.outbound_layer
                layer_name = layer.name

                input_args, input_kwargs = get_layer_input_tensors(
                    layer,
                    layer_outputs=layer_outputs,
                    layer_input_paths=layer_input_paths,
                    keras_module=keras_module
                )

                if layer in replacements:
                    layer = replacements[layer]

                if isinstance(layer, keras_module.models.Model):
                    output = layer.call(*input_args, **input_kwargs)
                else:
                    output = layer(*input_args, **input_kwargs)

                layer_outputs[layer_name] = output

            return prop_through_layer(depth=depth - 1, dirty=True)

    max_depth = max(list(model._nodes_by_depth.keys()))
    output = prop_through_layer(max_depth)
    new_submodel = keras_module.Model(inputs=model.inputs, outputs=output)
    new_submodel._name = model.name
    return new_submodel


def trace_input_indices(model, keras_module):
    '''
    trace_input_indices Get index of input nodes for each layer in model.
    If layers in model are shared across multiple models, they may have multiple
    disconnected inbound nodes. To specify which inbound nodes belong to the
    provided model, this method returns a mapping of layers to the index of their
    respective inbound node.

    Parameters
    ----------
    model : keras.models.Model
        Base model

    keras_module : Python Module
        Either the keras module or tf.keras module

    Returns
    -------
    Dict[str, int]
    Returns a mapping of layer names to the index of the inbound node associated with model

    '''

    def tracer(depth):
        if depth not in nodes_by_depth:
            return
        nodes = nodes_by_depth[depth]
        for node in nodes:
            out_layer = node.outbound_layer
            if isinstance(out_layer, keras_module.models.Model):
                # Is nested model, recurse
                sub_innode_indices = trace_input_indices(
                    out_layer, keras_module
                )
                for layer_name, idx in sub_innode_indices.items():
                    innode_indices[f'{out_layer.name}/{layer_name}'] = idx

            if node in out_layer._inbound_nodes:
                idx = out_layer._inbound_nodes.index(node)
                if out_layer.name in innode_indices and innode_indices[
                        out_layer.name] != idx:
                    # May occur if layer is shared between other layers in the same model
                    tru_logger.warning(
                        f'{out_layer.name}: input node conflict. orig: {innode_indices[out_layer.name]}, new: {idx}. Keeping original'
                    )
                if out_layer.name not in innode_indices:
                    innode_indices[out_layer.name] = idx
        tracer(depth + 1)

    try:
        innode_indices = {}
        nodes_by_depth = model._nodes_by_depth
        tracer(0)
    except AttributeError:
        tru_logger.warning(
            'The provided model is missing Keras graph nodes metadata.'
            'TruLens will assume layers are using their default input/output tensors.'
            "This should not cause issues unless the model's layers have multiple inputs."
        )
        return {}
    return innode_indices


def load_keras_model_from_handle(
    handle, orig_layer, keras_module, tfhub_module
):
    '''
    load_keras_model_from_handle Load the tensorflow hub KerasLayer as a keras Model.

    Parameters
    ----------
    handle : str
        TFHub handle

    orig_layer : KerasLayer

    keras_module : Python Module
        Either the keras module or tf.keras module

    tfhub_module : Python Module
        tensorflow_hub module. Used to resolve tfhub model paths

    Returns
    -------
    keras.models.Model

    '''
    module_path = tfhub_module.module_v2.resolve(handle)
    keras_model = keras_module.models.load_model(module_path)
    keras_model.set_weights(orig_layer.get_weights())
    keras_model._name = orig_layer.name
    return keras_model


def flatten_substitute_tfhub(model, keras_module, tfhub_module):
    '''
    flatten_substitute_tfhub Find all TFHub KerasLayers and
    replace them with a keras Model.

    Parameters
    ----------
    model : keras.models.Model
        The Keras model

    keras_module : Python Module
        Either the keras module or tf.keras module

    tfhub_module : Python Module
        tensorflow_hub module. Used to resolve tfhub model paths

    Returns
    -------
    keras.models.Model
    A model free of nested KerasLayers
    '''
    replacements = {}
    for layer in model.layers:
        try:
            layer_config = layer.get_config()
        except (NotImplementedError, TypeError):
            layer_config = None

        if tfhub_module and layer_config and 'handle' in layer_config:
            try:
                submodel = load_keras_model_from_handle(
                    layer_config['handle'], layer, keras_module, tfhub_module
                )
                rename_nested_layers(submodel, keras_module)
                submodel = flatten_substitute_tfhub(
                    submodel, keras_module, tfhub_module
                )
                replacements[layer] = submodel
            except (OSError, ValueError):
                # TODO: default to chainrule if keras layer substitution doesn't work
                tru_logger.warning(
                    f'Unable to substitute Tensorflow Hub model {layer.name} with Keras implementation.'
                    f'Its nested layers will be hidden.'
                )
        elif isinstance(layer, keras_module.models.Model):
            rename_nested_layers(layer, keras_module)
            submodel = flatten_substitute_tfhub(
                layer, keras_module, tfhub_module
            )
            if layer is not submodel:
                replacements[layer] = submodel

    has_nested_models = any(
        isinstance(layer, keras_module.models.Model) for layer in model.layers
    )
    if has_nested_models or replacements:
        model = perform_replacements(model, replacements, keras_module)
    return model
