import collections

from trulens.nn.backend import get_backend


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
        ret = {}
        prefix = [] if prefix is None else prefix

        if B.is_tensor(obj):
            return {obj.ref(): prefix}
        elif isinstance(obj, collections.abc.Mapping):
            for key, val in obj.items():
                ret.update(recurse_outputs(val, prefix + [key]))
        elif isinstance(obj, collections.abc.Iterable):
            for i, elem in enumerate(obj):
                ret.update(recurse_outputs(elem, prefix + [i]))
        else:
            print(type(obj))
        return ret

    def get_arg_path(layer, arg):
        if B.is_tensor(arg):
            return layer_outputs[layer.name][arg.ref()]
        elif isinstance(arg, collections.abc.Mapping):
            return {
                key: layer_outputs[layer.name][a.ref()]
                for key, a in arg.items()
            }
        elif isinstance(arg, collections.abc.Iterable):
            return [layer_outputs[layer.name][a.ref()] for a in arg]
        else:
            raise ValueError("Invalid argument found")

    layer_outputs = {}
    layer_input_paths = {}

    # Outputs of each layer
    # TODO: this may be cleaner with node traversal instead of layer traversal
    for layer in model.layers:
        layer_outputs[layer.name] = recurse_outputs(layer.output, [layer.name])

    # Path to inputs of each layer
    for layer in model.layers:
        out_nodes = set(layer.outbound_nodes)
        for out_node in out_nodes:
            next_layer = out_node.outbound_layer
            args = out_node.call_args
            kwargs = out_node.call_kwargs

            arg_paths = [get_arg_path(layer, arg) for arg in args]
            kwarg_paths = {
                key: get_arg_path(layer, arg) for key, arg in kwargs.items()
            }

            if next_layer.name not in layer_input_paths:
                layer_input_paths[next_layer.name] = {
                    "args": arg_paths,
                    "kwargs": kwarg_paths
                }
            else:
                layer_input_paths[next_layer.name]['args'].extend(arg_paths)
                layer_input_paths[next_layer.name]['kwargs'].update(kwarg_paths)
    return layer_input_paths


def get_layer_input_tensors(layer, layer_outputs, layer_input_paths):
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
                raise ValueError(f"Invalid part path {part} for object {out}")
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
            raise ValueError(f"Invalid arg path {arg}")

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
        if depth < 0:
            nodes = model._nodes_by_depth[0]
            return [layer_outputs[n.layer.name] for n in nodes]

        nodes = model._nodes_by_depth[depth]
        if not dirty and all(n.layer not in replacements for n in nodes):
            # no prior modifications and no replacements at this depth, continue on
            for n in nodes:
                layer_outputs[n.layer.name] = n.layer.get_output_at(-1)

            return prop_through_layer(depth=depth - 1, dirty=dirty)
        else:
            # is dirty or needs to perform replacement
            for n in nodes:
                layer = n.layer
                layer_fn = layer
                layer_name = layer.name

                input_args, input_kwargs = get_layer_input_tensors(
                    layer,
                    layer_outputs=layer_outputs,
                    layer_input_paths=layer_input_paths
                )

                if layer in replacements:
                    layer = replacements[layer]
                    layer_fn = layer.call

                output = layer_fn(*input_args, **input_kwargs)
                layer_outputs[layer_name] = output

            return prop_through_layer(depth=depth - 1, dirty=True)

    max_depth = max(list(model._nodes_by_depth.keys()))
    output = prop_through_layer(max_depth)

    new_model = keras_module.Model(inputs=model.inputs, outputs=output)
    return new_model


def trace_input_indices(model):
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
    
    Returns
    -------
    Dict[str, int]
    Returns a mapping of layer names to the index of the inbound node associated with model

    '''
    innode_indices = {}
    nodes_by_depth = model._nodes_by_depth

    def tracer(depth):
        if depth not in nodes_by_depth:
            return
        nodes = nodes_by_depth[depth]
        for node in nodes:
            out_layer = node.outbound_layer
            if hasattr(out_layer, "layers"):
                # Is nested model, recurse
                sub_innode_indices = trace_input_indices(out_layer)
                for layer_name, idx in sub_innode_indices.items():
                    innode_indices[f"{out_layer.name}/{layer_name}"] = idx

            if node in out_layer.inbound_nodes:
                idx = out_layer.inbound_nodes.index(node)
                if out_layer.name in innode_indices and innode_indices[
                        out_layer.name] != idx:
                    # May occur if layer is shared between other layers in the same model
                    print(
                        f"{out_layer.name}: input node conflict. orig: {innode_indices[out_layer.name]}, new: {idx}. Keeping original"
                    )
                if out_layer.name not in innode_indices:
                    innode_indices[out_layer.name] = idx
        tracer(depth + 1)

    tracer(0)
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

    # Unnest models from input layers
    while True:
        if not hasattr(keras_model, "layers"):
            break

        non_input_layers = []
        for layer in keras_model.layers:
            if not isinstance(layer, keras_module.layers.InputLayer):
                layer._name = f"{keras_model.name}/{layer.name}"
                non_input_layers.append(layer)

        if len(non_input_layers) != 1:
            break
        keras_model = non_input_layers[0]

    return keras_model


def replace_tfhub_layers(model, keras_module, tfhub_module):
    '''
    replace_tfhub_layers Find all TFHub KerasLayers and 
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
    layers = model.layers
    replacements = {}
    i = 0
    while i < len(layers):
        layer = layers[i]

        try:
            layer_config = layer.get_config()
        except NotImplementedError:
            layer_config = None

        if layer_config and "handle" in layer_config:
            try:
                keras_model = load_keras_model_from_handle(
                    layer_config['handle'], layer, keras_module, tfhub_module
                )
                keras_model = replace_tfhub_layers(
                    keras_model, keras_module, tfhub_module
                )
                replacements[layer] = keras_model
            except (OSError, ValueError):
                # TODO: default to chainrule if keras layer substitution doesn't work
                print(
                    f"Unable to substitute Tensorflow model {layer.name} with Keras implementation"
                )
        i += 1

    if len(replacements) > 0:
        return perform_replacements(model, replacements, keras_module)
    return model
