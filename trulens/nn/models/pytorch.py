from collections import OrderedDict
from functools import partial
from logging import LogRecord
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from trulens.nn.backend import get_backend
from trulens.nn.backend.pytorch_backend import pytorch
from trulens.nn.backend.pytorch_backend.pytorch import Tensor, grace
from trulens.nn.models._model_base import ModelWrapper
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import LogitCut
from trulens.nn.slices import OutputCut
from trulens.utils import tru_logger
from trulens.utils.typing import ArgsLike
from trulens.utils.typing import as_args
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import DataLike
from trulens.utils.typing import InterventionLike
from trulens.utils.typing import ModelInputs


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
        device=None
    ):
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
        if input_dtype is None:
            tru_logger.debug(
                "Input dtype was not passed in. Defaulting to `torch.float32`."
            )
            input_dtype = torch.float32
        if input_shape is None:
            raise ValueError(
                'pytorch model wrapper must pass the input_shape parameter'
            )
        model.eval()

        if device is None:
            device = pytorch.get_default_device()

        pytorch.set_default_device(device)
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

        if len(self._tensors) == 0:
            tru_logger.warn(
                "model has no visible components, you will not be able to specify cuts"
            )

        # Check to see if this model outputs probits or logits.
        if len(self._tensors) > 0 and isinstance(self._tensors[-1],
                                                 torch.nn.Softmax):
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
                        ('{}_{}'.format(name, cr_layer_name), cr_layer_mod)
                    )
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
                    self._logit_layer, cut.anchor
                )
            )

        elif isinstance(cut.name, DATA_CONTAINER_TYPE):
            for name in cut.name:
                names_and_anchors.append((name, cut.anchor))

        elif not (isinstance(cut, OutputCut) or isinstance(cut, InputCut)):
            names_and_anchors.append((cut.name, cut.anchor))

    def _extract_outputs_from_hooks(
        self, cut, hooks, output, model_inputs, return_tensor
    ):

        B = get_backend()

        return_output = None

        if isinstance(cut, OutputCut):
            return_output = output

        elif isinstance(cut, InputCut):
            # TODO(piotrm): Figure out whether kwarg order is consistent.
            return_output = tuple(model_inputs.args
                                 ) + tuple(model_inputs.kwargs.values())

        elif isinstance(cut, LogitCut):
            return_output = hooks['logits' if self.
                                  _logit_layer is None else self._logit_layer]

        elif isinstance(cut.name, DATA_CONTAINER_TYPE):
            return_output = [hooks[name] for name in cut.name]

        else:
            return_output = hooks[cut.name]

        return_output = ModelWrapper._flatten(return_output)

        if return_tensor:
            return return_output
        else:
            return ModelWrapper._nested_apply(return_output, B.as_array)

    def _to_tensor(self, x):
        # Convert `x` to a tensor on `self.device`. Note that layer input can be
        # a nested DATA_CONTAINER_TYPE.
        B = get_backend()
        if isinstance(x, np.ndarray) or (len(x) > 0 and
                                         isinstance(x[0], np.ndarray)):
            x = ModelWrapper._nested_apply(
                x, partial(B.as_tensor, device=self.device)
            )

        elif isinstance(x, DATA_CONTAINER_TYPE):
            x = [self._to_tensor(x_i) for x_i in x]

        else:
            x = ModelWrapper._nested_apply(x, lambda x: x.to(self.device))

        return x

    def fprop(
        self,
        model_args: ArgsLike,
        model_kwargs: Dict[str, DataLike] = {},
        doi_cut: Optional[Cut] = None,
        to_cut: Optional[Cut] = None,
        attribution_cut: Optional[Cut] = None,
        intervention: InterventionLike = None,
        return_tensor: bool = False,
        input_timestep: Optional[int] = None
    ) -> Union[List[Tensor], List[np.ndarray]]:
        """
        fprop Forward propagate the model.

        Parameters
        ----------
        model_args, model_kwargs: 
            The args and kwargs given to the call method of a model. This should
            represent the instances to obtain attributions for, assumed to be a
            *batched* input. if `self.model` supports evaluation on *data
            tensors*, the appropriate tensor type may be used (e.g., Pytorch
            models may accept Pytorch tensors in additon to `np.ndarray`s). The
            shape of the inputs must match the input shape of `self.model`. 
        doi_cut: Cut, optional
            The Cut from which to begin propagation. The shape of `intervention`
            must match the input shape of this layer. This is usually used to
            apply distributions of interest (DoI)
        to_cut : Cut, optional
            The Cut to return output activation tensors for. If `None`, assumed
            to be just the final layer. By default None
        attribution_cut : Cut, optional
            An Cut to return activation tensors for. If `None` attributions
            layer output is not returned.
        intervention : ArgsLike (for non-InputCut DoIs) or
            ModelInputs (for InputCut DoIs) Tensor(s) to propagate through the
            model. If an intervention is ArgsLike for InputCut, we assume there
            are no kwargs.
        input_timestep: int, optional
            Timestep to apply to the DoI if using an RNN

        Returns
        -------
        (list of backend.Tensor or list of np.ndarray)
            A list of output activations are returned, keeping the same type as
            the input. If `attribution_cut` is supplied, also return the cut
            activations.
        """

        B = get_backend()

        doi_cut, to_cut, intervention, model_inputs = ModelWrapper._fprop_defaults(
            self,
            model_args=model_args,
            model_kwargs=model_kwargs,
            doi_cut=doi_cut,
            to_cut=to_cut,
            intervention=intervention
        )

        model_inputs = model_inputs.map(self._to_tensor)
        intervention = intervention.map(self._to_tensor)

        if isinstance(doi_cut, InputCut):
            # all of the necessary logic here has been factored out into
            # _fprop_defaults
            pass

        else:  # doi_cut != InputCut
            # Tile model inputs so that batch dim at cut matches intervention
            # batch dim.

            expected_dim = model_inputs.first().shape[0]
            doi_resolution = int(intervention.first().shape[0] // expected_dim)

            def tile_val(val):
                """Tile the given value if expected_dim matches val's first
                dimension. Otherwise return original val unchanged."""

                if val.shape[0] != expected_dim:
                    tru_logger.warn(
                        f"Value {val} of shape {val.shape} is assumed to not be "
                        f"batchable due to its shape not matching prior batchable "
                        f"inputs of shape ({expected_dim},...). If this is "
                        f"incorrect, make sure its first dimension matches prior "
                        f"batchable inputs."
                    )
                    return val

                tile_shape = [1 for _ in range(len(val.shape))]
                tile_shape[0] = doi_resolution
                repeat_shape = tuple(tile_shape)

                with grace(device=self.device):
                    # likely place where memory issues might arise

                    if isinstance(val, np.ndarray):
                        return np.tile(val, repeat_shape)
                    elif torch.is_tensor(val):
                        return val.repeat(repeat_shape)
                    else:
                        raise ValueError(
                            f"unhandled tensor type {val.__class__.__name__}"
                        )

            # tile args and kwargs if necessary
            model_inputs = model_inputs.map(tile_val)

        if attribution_cut is not None:
            # Specify that we want to preserve gradient information.

            def enable_grad(t: torch.Tensor):
                if torch.is_floating_point(t):
                    t.requires_grad_(True)
                else:
                    if isinstance(attribution_cut, InputCut):
                        raise ValueError(
                            f"Requested tensors for attribution_cut=InputCut() but it contains a non-differentiable tensor of type {t.dtype}. You may need to provide an attribution_cut further down in the model where floating-point values first arise."
                        )
                    else:
                        # Could be a warning here but then we'd see a lot of warnings in NLP models.
                        pass

            intervention.foreach(lambda v: v.requires_grad_(True))
            model_inputs.foreach(enable_grad)

        # Set up the intervention hookfn if we are starting from an intermediate
        # layer.

        if not isinstance(doi_cut, InputCut):
            # Interventions only allowed onto one layer (see FIXME below.)
            assert len(intervention) == 1

            # Define the hookfn.
            counter = 0

            def intervene_hookfn(self, inpt, outpt):
                nonlocal counter

                if input_timestep is None or input_timestep == counter:
                    # FIXME: generalize to multi-input layers. Currently can
                    #   only intervene on one layer.
                    inpt = inpt[0] if len(inpt) == 1 else inpt
                    if doi_cut.anchor == 'in':
                        ModelWrapper._nested_assign(inpt, intervention.first())
                    else:
                        ModelWrapper._nested_assign(outpt, intervention.first())

                counter += 1

            # Register according to the anchor.
            if doi_cut.anchor == 'in':
                in_handle = (
                    self._get_layer(doi_cut.name).register_forward_pre_hook(
                        partial(intervene_hookfn, outpt=None)
                    )
                )
            else:
                in_handle = (
                    self._get_layer(doi_cut.name
                                   ).register_forward_hook(intervene_hookfn)
                )

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
                            inpt, B.as_array
                        )
                    else:
                        outpt = outpt[0] if isinstance(outpt, tuple) else outpt
                        hooks[layer_name] = ModelWrapper._nested_apply(
                            outpt, B.as_array
                        )

            return hookfn

        handles = [
            self._get_layer(name).register_forward_hook(
                get_hookfn(name, anchor)
            ) for name, anchor in names_and_anchors if name is not None
        ]

        with grace(device=self.device):
            # Run the network.
            self._model.eval()  # needed for determinism sometimes
            output = model_inputs.call_on(self._model)

        if isinstance(output, tuple):
            output = output[0]

        if not isinstance(doi_cut, InputCut):
            # Clean up in handle.
            in_handle.remove()

        # Clean up out handles.
        for handle in handles:
            handle.remove()

        extract_args = dict(
            hooks=hooks,
            output=output,
            model_inputs=model_inputs,
            return_tensor=return_tensor
        )

        if attribution_cut:
            return [
                self._extract_outputs_from_hooks(cut=to_cut, **extract_args),
                self._extract_outputs_from_hooks(
                    cut=attribution_cut, **extract_args
                )
            ]
        else:
            return self._extract_outputs_from_hooks(cut=to_cut, **extract_args)

    def qoi_bprop(
        self,
        qoi: QoI,
        model_args: ArgsLike,
        model_kwargs: Dict[str, DataLike] = {},
        doi_cut: Optional[Cut] = None,
        to_cut: Optional[Cut] = None,
        attribution_cut: Optional[Cut] = None,
        intervention: InterventionLike = None
    ):
        """
        qoi_bprop Run the model from the from_layer to the qoi layer
            and give the gradients w.r.t `attribution_cut`

        Parameters
        ----------
        qoi: a Quantity of Interest
            This method will accumulate all gradients of the qoi w.r.t
            `attribution_cut`.
        model_args, model_kwargs: 
            The args and kwargs given to the call method of a model. This should
            represent the instances to obtain attributions for, assumed to be a
            *batched* input. if `self.model` supports evaluation on *data
            tensors*, the  appropriate tensor type may be used (e.g., Pytorch
            models may accept Pytorch tensors in additon to `np.ndarray`s). The
            shape of the inputs must match the input shape of `self.model`. 
        doi_cut: Cut, 
            if `doi_cut` is None, this refers to the InputCut. Cut from which to
            begin propagation. The shape of `intervention` must match the output
            shape of this layer.
        attribution_cut: Cut, optional
            if `attribution_cut` is None, this refers to the InputCut. The Cut
            in which attribution will be calculated. This is generally taken
            from the attribution slyce's attribution_cut.
        to_cut: Cut, optional
            if `to_cut` is None, this refers to the OutputCut. The Cut in which
            qoi will be calculated. This is generally taken from the attribution
            slyce's to_cut.
        intervention: InterventionLike
            Input tensor to propagate through the model. If an np.array, will be
            converted to a tensor on the same device as the model.

        Returns
        -------
        (backend.Tensor or np.ndarray)
            the gradients of `qoi` w.r.t. `attribution_cut`, keeping same type
            as the input.
        """
        B = get_backend()

        doi_cut, to_cut, attribution_cut = self._qoi_bprop_defaults(
            doi_cut=doi_cut, to_cut=to_cut, attribution_cut=attribution_cut
        )

        y, zs = self.fprop(
            model_args=model_args,
            model_kwargs=model_kwargs,
            doi_cut=doi_cut,
            to_cut=to_cut,
            attribution_cut=attribution_cut,
            intervention=intervention,
            return_tensor=True
        )

        y = to_cut.access_layer(y)
        grads_list = []

        def scalarize(t: torch.Tensor) -> torch.tensor:
            if len(t.shape) > 1 and np.array(t.shape).prod() != 1:
                # Adding warning here only if there is more than 1 dimension
                # being summed. If there is only 1 dim, its likely the batching
                # dimension so sum there is probably expected.
                tru_logger.warn(
                    f"Attribution tensor is not scalar (it is of shape {t.shape} and will be summed. This may not be your intention."
                )

            return B.sum(t)

        try:
            for z in zs:
                z_flat = ModelWrapper._flatten(z)
                qoi_out = qoi(y)

                # TODO(piotrm): this sum is a source of much bugs for me when using
                # attributions. If one wants a specific QoI, the sum hides the bugs
                # in the definition of that QoI. It might be better to give an error
                # when QoI is not a scalar.
                with grace(device=self.device):
                    # Common place where memory issues arise.

                    grads_flat = [
                        B.gradient(scalarize(q), z_flat) for q in qoi_out
                    ] if isinstance(qoi_out, DATA_CONTAINER_TYPE) else B.gradient(scalarize(qoi_out), z_flat)

                grads = [
                    ModelWrapper._unflatten(g, z, count=[0]) for g in grads_flat
                ] if isinstance(qoi_out,
                                DATA_CONTAINER_TYPE) else ModelWrapper._unflatten(
                                    grads_flat, z, count=[0]
                                )

                grads = [attribution_cut.access_layer(g) for g in grads
                        ] if isinstance(qoi_out, DATA_CONTAINER_TYPE
                                    ) else attribution_cut.access_layer(grads)

                grads = [B.as_array(g) for g in grads
                        ] if isinstance(qoi_out,
                                        DATA_CONTAINER_TYPE) else B.as_array(grads)

        except RuntimeError as e:
            if "cudnn RNN backward can only be called in training mode" in str(e):
                raise RuntimeError(
                    "Cannot get deterministic gradients from RNN's with cudnn. See more about this issue here: https://github.com/pytorch/captum/issues/564 .\n"
                    "Consider setting 'torch.backends.cudnn.enabled = False' for now."
                )
            raise e


        del y  # TODO: garbage collection

        return grads_list
        # NOTE(piotrm): commented out the below to have more consistent output types/shapes
        
        # return grads_list[0] if len(grads_list) == 1 else grads_list

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
        B = get_backend()
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
