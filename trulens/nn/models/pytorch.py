from collections import OrderedDict
from functools import partial

import numpy as np
import torch

from trulens.nn import backend as B
from trulens.nn.slices import InputCut, OutputCut, LogitCut
from trulens.nn.models._model_base import ModelWrapper, DATA_CONTAINER_TYPE


class PytorchModelWrapper(ModelWrapper):
    """
    Model wrapper that exposes the internal components
    of Pytorch nn.Module objects.
    """

    def __init__(
            self,
            model,
            input_shape,
            input_dtype=torch.float32,
            logit_layer=None,
            device=None):
        """
        __init__ Constructor

        Parameters
        ----------
        model : pytorch.nn.Module
            Pytorch model implemented as nn.Module
        input_shape: tuple
            The shape of the input without the batch dimension.
        input_dtype: torch.dtype
            The dtype of the input.
        logit_layer: str
            The name of the logit layer. If not supplied, it will assume any 
            layer named 'logits' is the logit layer.
        device : string, optional
            device on which to run model, by default None
        """
        model.eval()
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        model.to(self.device)
        self._model = model
        self._input_shape = input_shape
        self._input_dtype = input_dtype
        self._logit_layer = logit_layer

        layers = OrderedDict(PytorchModelWrapper._get_model_layers(model))
        self._layers = layers
        self._layernames = list(layers.keys())
        self._tensors = list(layers.values())

        # Check to see if this model outputs probits or logits.
        if isinstance(self._tensors[-1], torch.nn.Softmax):
            self._gives_logits = False
        else:
            self._gives_logits = True

    def print_layer_names(self):
        for name in self._layernames:
            print(f'\'{name}\':\t{self._layers[name]}')

    @staticmethod
    def _get_model_layers(model):
        """
        _get_model_layers Internal method to get layers from Pytorch module

        Parameters
        ----------
        model : pytorch.nn.Module
            Target model

        Returns
        -------
        list of (name, module) tuples
            Obtained by recursively calling get_named_children()
        """
        r_layers = []
        for name, mod in model.named_children():
            if len(list(mod.named_children())) == 0:
                r_layers.append((name, mod))
            else:
                cr_layers = PytorchModelWrapper._get_model_layers(mod)
                for cr_layer_name, cr_layer_mod in cr_layers:
                    r_layers.append(
                        ('{}_{}'.format(name, cr_layer_name), cr_layer_mod))
        return r_layers

    def _get_layer(self, name):
        """
        get_layer Return identified layer

        Parameters
        ----------
        name : int or string
            If given as int, return layer at that index. If given
            as a string, return the layer with the corresponding name.

        Returns
        -------
        pytorch.nn.Module
            Layer obtained from model's named_children method.

        Raises
        ------
        ValueError
            No layer with given name identifier
        ValueError
            Layer index out of bounds
        """
        if isinstance(name, str):
            if not name in self._layernames:
                raise ValueError('no such layer tensor:', name)
            return self._layers[name]
        elif isinstance(name, int):
            if len(self._layers) <= name:
                raise ValueError('layer index out of bounds:', name)
            return self._layers[self._layernames[name]]
        elif isinstance(name, DATA_CONTAINER_TYPE):
            return [self._get_layer(n) for n in name]
        else:
            return name

    def _add_cut_name_and_anchor(self, cut, names_and_anchors):
        if isinstance(cut, LogitCut):
            names_and_anchors.append(
                (
                    'logits' if self._logit_layer is None else
                    self._logit_layer, cut.anchor))

        elif isinstance(cut.name, DATA_CONTAINER_TYPE):
            for name in cut.name:
                names_and_anchors.append((name, cut.anchor))

        elif not (isinstance(cut, OutputCut) or isinstance(cut, InputCut)):
            names_and_anchors.append((cut.name, cut.anchor))

    def _extract_outputs_from_hooks(
            self, cut, hooks, output, model_input, return_tensor):
        if isinstance(cut, OutputCut):
            return (
                ModelWrapper._flatten(output)
                if return_tensor else ModelWrapper._nested_apply(
                    ModelWrapper._flatten(output), B.as_array))

        elif isinstance(cut, InputCut):
            return (
                ModelWrapper._flatten(model_input)
                if return_tensor else ModelWrapper._nested_apply(
                    ModelWrapper._flatten(model_input), B.as_array))

        elif isinstance(cut, LogitCut):
            y = hooks['logits' if self._logit_layer is None else self.
                      _logit_layer]
            return (
                ModelWrapper._flatten(y)
                if return_tensor else ModelWrapper._nested_apply(
                    ModelWrapper._flatten(y), B.as_array))

        elif isinstance(cut.name, DATA_CONTAINER_TYPE):
            zs = [hooks[name] for name in cut.name]
            return (
                ModelWrapper._flatten(zs)
                if return_tensor else ModelWrapper._nested_apply(
                    ModelWrapper._flatten(zs), B.as_array))

        else:
            z = hooks[cut.name]
            return (
                ModelWrapper._flatten(z)
                if return_tensor else ModelWrapper._nested_apply(
                    ModelWrapper._flatten(z), B.as_array))

    def _to_tensor(self, x):
        # Convert `x` to a tensor on `self.device`. Note that layer input can be
        # a nested DATA_CONTAINER_TYPE.
        if isinstance(x, np.ndarray) or isinstance(x[0], np.ndarray):
            x = ModelWrapper._nested_apply(
                x, partial(B.as_tensor, device=self.device))

        elif isinstance(x, DATA_CONTAINER_TYPE):
            x = [self._to_tensor(x_i) for x_i in x]

        else:
            x = ModelWrapper._nested_apply(x, lambda x: x.to(self.device))

        return x

    def fprop(
            self,
            model_args,
            model_kwargs={},
            doi_cut=None,
            to_cut=None,
            attribution_cut=None,
            intervention=None,
            return_tensor=False,
            input_timestep=None):
        """
        fprop Forward propagate the model

        Parameters
        ----------
        model_args, model_kwargs: 
            The args and kwargs given to the call method of a model.
            This should represent the instances to obtain attributions for, 
            assumed to be a *batched* input. if `self.model` supports evaluation 
            on *data tensors*, the  appropriate tensor type may be used (e.g.,
            Pytorch models may accept Pytorch tensors in additon to
            `np.ndarray`s). The shape of the inputs must match the input shape
            of `self.model`. 
        doi_cut: Cut, optional
            The Cut from which to begin propagation. The shape of `intervention`
            must match the input shape of this layer. This is usually used to 
            apply distributions of interest (DoI)
        to_cut : Cut, optional
            The Cut to return output activation tensors for. If `None`,
            assumed to be just the final layer. By default None
        attribution_cut : Cut, optional
            An Cut to return activation tensors for. If `None` 
            attributions layer output is not returned.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array, 
            will be converted to a tensor on the same device as the model.
        input_timestep: int, optional
            Specifies a specific timestep to apply the DoI if using an RNN

        Returns
        -------
        (list of backend.Tensor or np.ndarray)
            A list of output activations are returned, keeping the same type as
            the input. If `attribution_cut` is supplied, also return the cut 
            activations.
        """

        if doi_cut is None:
            doi_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        model_args = self._to_tensor(model_args)

        if intervention is None:
            intervention = model_args

        intervention = intervention if isinstance(
            intervention, DATA_CONTAINER_TYPE) else [intervention]
        intervention = self._to_tensor(intervention)

        if (isinstance(doi_cut, InputCut)):
            model_args = intervention

        else:
            doi_repeated_batch_size = intervention[0].shape[0]
            batched_model_args = []

            for val in model_args:
                doi_resolution = int(doi_repeated_batch_size / val.shape[0])
                tile_shape = [1 for _ in range(len(val.shape))]
                tile_shape[0] = doi_resolution
                repeat_shape = tuple(tile_shape)

                if isinstance(val, np.ndarray):
                    val = np.tile(val, repeat_shape)

                elif torch.is_tensor(val):
                    val = val.repeat(repeat_shape)

                batched_model_args.append(val)

            model_args = batched_model_args

        if (attribution_cut is not None):
            # Specify that we want to preserve gradient information.
            intervention = ModelWrapper._nested_apply(
                intervention,
                lambda intervention: intervention.requires_grad_(True))
            model_args = ModelWrapper._nested_apply(
                model_args, lambda model_args: model_args.requires_grad_(True))

        # Set up the intervention hookfn if we are starting from an intermediate
        # layer.
        if not isinstance(doi_cut, InputCut):
            # Define the hookfn.
            counter = 0

            def intervene_hookfn(self, inpt, outpt):
                nonlocal counter, input_timestep, doi_cut, intervention

                if input_timestep is None or input_timestep == counter:
                    # FIXME: generalize to multi-input layers. Currently can 
                    #   only intervene on one layer.
                    inpt = inpt[0] if len(inpt) == 1 else inpt
                    if doi_cut.anchor == 'in':
                        ModelWrapper._nested_assign(inpt, intervention[0])
                    else:
                        ModelWrapper._nested_assign(outpt, intervention[0])

                counter += 1

            # Register according to the anchor.
            if doi_cut.anchor == 'in':
                in_handle = (
                    self._get_layer(doi_cut.name).register_forward_pre_hook(
                        partial(intervene_hookfn, outpt=None)))
            else:
                in_handle = (
                    self._get_layer(
                        doi_cut.name).register_forward_hook(intervene_hookfn))

        # Collect the names and anchors of the layers we want to return.
        names_and_anchors = []

        self._add_cut_name_and_anchor(to_cut, names_and_anchors)

        if attribution_cut:
            self._add_cut_name_and_anchor(attribution_cut, names_and_anchors)

        # Create hookfns to extract the results from the specified layers.
        hooks = {}

        def get_hookfn(layer_name, anchor):

            def hookfn(self, inpt, outpt):
                nonlocal hooks, layer_name, anchor
                # FIXME: generalize to multi-input layers
                inpt = inpt[0] if len(inpt) == 1 else inpt

                if return_tensor:
                    if anchor == 'in':
                        hooks[layer_name] = inpt
                    else:
                        # FIXME : will not work for multibranch outputs
                        # needed to ignore hidden states of RNNs
                        outpt = outpt[0] if isinstance(outpt, tuple) else outpt
                        hooks[layer_name] = outpt

                else:
                    if anchor == 'in':
                        hooks[layer_name] = ModelWrapper._nested_apply(
                            inpt, B.as_array)
                    else:
                        outpt = outpt[0] if isinstance(outpt, tuple) else outpt
                        hooks[layer_name] = ModelWrapper._nested_apply(
                            outpt, B.as_array)

            return hookfn

        handles = [
            self._get_layer(name).register_forward_hook(
                get_hookfn(name, anchor))
            for name, anchor in names_and_anchors
            if name is not None
        ]
        # Run the network.
        output = self._model(*model_args, *model_kwargs)
        if isinstance(output, tuple):
            output = output[0]

        if not isinstance(doi_cut, InputCut):
            # Clean up in handle.
            in_handle.remove()

        # Clean up out handles.
        for handle in handles:
            handle.remove()

        if attribution_cut:
            return [
                self._extract_outputs_from_hooks(
                    to_cut, hooks, output, model_args, return_tensor),
                self._extract_outputs_from_hooks(
                    attribution_cut, hooks, output, model_args, return_tensor)
            ]
        else:
            return self._extract_outputs_from_hooks(
                to_cut, hooks, output, model_args, return_tensor)

    def qoi_bprop(
            self,
            qoi,
            model_args,
            model_kwargs={},
            doi_cut=None,
            to_cut=None,
            attribution_cut=None,
            intervention=None):
        """
        qoi_bprop Run the model from the from_layer to the qoi layer
            and give the gradients w.r.t `attribution_cut`

        Parameters
        ----------
        model_args, model_kwargs: 
            The args and kwargs given to the call method of a model.
            This should represent the instances to obtain attributions for, 
            assumed to be a *batched* input. if `self.model` supports evaluation 
            on *data tensors*, the  appropriate tensor type may be used (e.g.,
            Pytorch models may accept Pytorch tensors in additon to 
            `np.ndarray`s). The shape of the inputs must match the input shape
            of `self.model`. 
        
        qoi: a Quantity of Interest
            This method will accumulate all gradients of the qoi w.r.t
            `attribution_cut`.
        doi_cut: Cut, 
            if `doi_cut` is None, this refers to the InputCut.
            Cut from which to begin propagation. The shape of `intervention`
            must match the output shape of this layer.
        attribution_cut: Cut, optional
            if `attribution_cut` is None, this refers to the InputCut.
            The Cut in which attribution will be calculated. This is generally
            taken from the attribution slyce's attribution_cut.
        to_cut: Cut, optional
            if `to_cut` is None, this refers to the OutputCut.
            The Cut in which qoi will be calculated. This is generally
            taken from the attribution slyce's to_cut.
        intervention : backend.Tensor or np.array
            Input tensor to propagate through the model. If an np.array,
            will be converted to a tensor on the same device as the model.

        Returns
        -------
        (backend.Tensor or np.ndarray)
            the gradients of `qoi` w.r.t. `attribution_cut`, keeping same type 
            as the input.
        """
        if attribution_cut is None:
            attribution_cut = InputCut()
        if to_cut is None:
            to_cut = OutputCut()

        y, zs = self.fprop(
            model_args,
            model_kwargs,
            doi_cut=doi_cut if doi_cut else InputCut(),
            to_cut=to_cut,
            attribution_cut=attribution_cut,
            intervention=intervention,
            return_tensor=True)

        y = to_cut.access_layer(y)
        grads_list = []
        for z in zs:
            z_flat = ModelWrapper._flatten(z)
            qoi_out = qoi(y)

            grads_flat = [
                B.gradient(B.sum(q), z_flat) for q in qoi_out
            ] if isinstance(qoi_out, DATA_CONTAINER_TYPE) else B.gradient(
                B.sum(qoi_out), z_flat)

            grads = [
                ModelWrapper._unflatten(g, z, count=[0]) for g in grads_flat
            ] if isinstance(
                qoi_out, DATA_CONTAINER_TYPE) else ModelWrapper._unflatten(
                    grads_flat, z, count=[0])

            grads = [
                attribution_cut.access_layer(g) for g in grads
            ] if isinstance(
                qoi_out,
                DATA_CONTAINER_TYPE) else attribution_cut.access_layer(grads)

            grads = [B.as_array(g) for g in grads] if isinstance(
                qoi_out, DATA_CONTAINER_TYPE) else B.as_array(grads)
            
            grads_list.append(grads)

        del y  # TODO: garbage collection

        return grads_list[0] if len(grads_list) == 1 else grads_list

    def probits(self, x):
        """
        probits Return probability outputs of the model

        This method assumes that if the final operation of
        the nn.Module given to the constructor is not
        torch.nn.Softmax, then the model returns logits as-is.

        Parameters
        ----------
        x : backend.Tensor or np.array
            Input point

        Returns
        -------
        backend.Tensor
            If the model is found to return logits (see above),
            then the result is obtained by applying torch.nn.Softmax.
            Otherwise, the model's output is simply returned.
        """
        output = self.fprop(x)
        if self._gives_logits:
            if len(output.shape) > 2:
                return B.softmax(output, axis=-1)
            else:
                return B.sigmoid(output)
        else:
            return output

    def logits(self, x):
        """
        logits Return outputs of model in log space

        This method assumes that if the final operation of
        the nn.Module given to the constructor is not
        torch.nn.Softmax, then the model returns logits as-is.

        Parameters
        ----------
        x : backend.Tensor or np.array
            Input point

        Returns
        -------
        backend.Tensor
            If the model is found to return logits (see above),
            then the result is obtained by fprop.
            Otherwise, the output is obtained by registering a
            callback on the model's penultimate layer prior to
            running fprop.
        """
        if self._gives_logits:
            return self.fprop(x)
        else:
            hook_in = None

            def hookfn(self, input, output):
                nonlocal hook_in
                hook_in = input

            logits = self._layers[-1]
            handle = logits.register_forward_hook(hookfn)
            if isinstance(x, np.ndarray):
                x = torch.Tensor(x)
            self._model(x.to(self.device))
            handle.remove()
            return hook_in
