from collections import Counter
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
from trulens.nn.backend import get_backend
from trulens.nn.backend.pytorch_backend import pytorch
from trulens.nn.backend.pytorch_backend.pytorch import memory_suggestions
from trulens.nn.backend.pytorch_backend.pytorch import Tensor
from trulens.nn.models._model_base import ModelWrapper
from trulens.nn.quantities import QoI
from trulens.nn.slices import Cut
from trulens.nn.slices import InputCut
from trulens.nn.slices import LogitCut
from trulens.nn.slices import OutputCut
from trulens.utils import tru_logger
from trulens.utils.typing import DATA_CONTAINER_TYPE
from trulens.utils.typing import Inputs
from trulens.utils.typing import many_of_om
from trulens.utils.typing import ModelInputs
from trulens.utils.typing import nested_map
from trulens.utils.typing import om_of_many
from trulens.utils.typing import Outputs
from trulens.utils.typing import TensorArgs
from trulens.utils.typing import TensorLike


class PytorchModelWrapper(ModelWrapper):
    """
    Model wrapper that exposes the internal components
    of Pytorch nn.Module objects.
    """

    def __init__(
        self,
        model,
        *,
        logit_layer=None,
        device=None,
        force_eval=True,
        **kwargs
    ):
        """
        __init__ Constructor

        Parameters
        ----------
        model : pytorch.nn.Module
            Pytorch model implemented as nn.Module
        logit_layer: str
            The name of the logit layer. If not supplied, it will assume any
            layer named 'logits' is the logit layer.
        device : string, optional
            device on which to run model, by default None
        force_eval : bool, optional
            If True, will call model.eval() to ensure determinism. Otherwise, keeps current model state, by default True

        """

        if 'input_shape' in kwargs:
            tru_logger.deprecate(
                f'PytorchModelWrapper: input_shape parameter is no longer used and will be removed in the future'
            )
            del kwargs['input_shape']
        if 'input_dtype' in kwargs:
            tru_logger.deprecate(
                f'PytorchModelWrapper: input_dtype parameter is no longer used and will be removed in the future'
            )
            del kwargs['input_dtype']

        super().__init__(model, **kwargs)
        # sets self._model, issues cross-backend messages
        self.force_eval = force_eval
        if self.force_eval:
            model.eval()

        if device is None:
            try:
                device_counter = Counter(
                    [param.get_device() for param in model.parameters()]
                )
                device = torch.device(device_counter.most_common()[0][0])
            except:
                device = pytorch.get_default_device()

        pytorch.set_default_device(device)

        self.device = device
        model.to(self.device)

        self._logit_layer = logit_layer

        layers = OrderedDict(PytorchModelWrapper._get_model_layers(model))
        self._layers = layers
        self._layernames = list(layers.keys())
        self._tensors = list(layers.values())

        if len(self._tensors) == 0:
            tru_logger.warning(
                'model has no visible components, you will not be able to specify cuts'
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

    def _extract_outputs_from_hooks(self, cut, hooks, output,
                                    model_inputs) -> Inputs[TensorLike]:

        return_output = None

        def _get_hook_val(k):
            if k not in hooks:
                # TODO: incoporate some more info in this error. Some of these
                # prints might be useful for this error.

                # self.print_layer_names()
                # print(hooks.keys())

                # TODO: create a new exception type for this so it can be caught
                # by downstream users better.

                # TODO: similar messages for other backends.

                raise ValueError(
                    f'Could not get values for layer {k}. Is it evaluated when computing doi cut from input cut?'
                )
            return hooks[k]

        if isinstance(cut, OutputCut):
            return_output = many_of_om(output)

        elif isinstance(cut, InputCut):
            return_output = list(model_inputs.args
                                ) + list(model_inputs.kwargs.values())

        elif isinstance(cut, LogitCut):
            return_output = many_of_om(
                hooks['logits' if self._logit_layer is None else self.
                      _logit_layer]
            )

        elif isinstance(cut.name, DATA_CONTAINER_TYPE):
            return_output = [_get_hook_val(name) for name in cut.name]

        else:
            return_output = many_of_om(_get_hook_val(cut.name))

        return return_output

    def _to_tensor(self, x):
        # Convert `x` to a tensor on `self.device`. Note that layer input can be
        # a nested DATA_CONTAINER_TYPE.
        B = get_backend()
        if isinstance(x, np.ndarray) or (len(x) > 0 and
                                         isinstance(x[0], np.ndarray)):
            x = nested_map(x, partial(B.as_tensor, device=self.device))

        elif isinstance(x, DATA_CONTAINER_TYPE):
            x = [self._to_tensor(x_i) for x_i in x]

        else:
            x = nested_map(x, lambda x: x.to(self.device))

        return x

    def _fprop(
        self,
        model_inputs: ModelInputs,
        doi_cut: Cut,
        to_cut: Cut,
        attribution_cut: Cut,
        intervention: TensorArgs,
        input_timestep: Optional[int] = None
    ) -> Tuple[Outputs[TensorLike], Outputs[TensorLike]]:
        """
        See ModelWrapper.fprop .

        Backend-Specific Parameters
        ----------
        input_timestep: int, optional
            Timestep to apply to the DoI if using an RNN
        """

        B = get_backend()

        # This method operates on backend tensors.
        intervention = intervention.map(B.as_tensor)

        # TODO: generalize the condition to include Cut objects that start at the beginning. Until then, clone the model args to avoid mutations (see MLNN-229)
        if isinstance(doi_cut, InputCut):
            model_inputs = intervention
        else:
            model_inputs = model_inputs.map(B.as_tensor)
            model_inputs = model_inputs.map(B.clone)

        if attribution_cut is not None:
            # Specify that we want to preserve gradient information.

            def enable_grad(t: torch.Tensor):
                if torch.is_floating_point(t):
                    t.requires_grad_(True)
                else:
                    if isinstance(attribution_cut, InputCut):
                        raise ValueError(
                            f'Requested tensors for attribution_cut=InputCut() but it contains a '
                            f'non-differentiable tensor of type {t.dtype}. You may need to provide '
                            f'an attribution_cut further down in the model where floating-point '
                            f'values first arise.'
                        )
                    else:
                        # Could be a warning here but then we'd see a lot of warnings in NLP models.
                        pass

            intervention.foreach(lambda v: v.requires_grad_(True))
            model_inputs.foreach(enable_grad)

        # Set up the intervention hookfn if we are starting from an intermediate
        # layer. These hooks replace the activations of the model at doi_cut
        # with what is given by intervention. This can cause some confusion as
        # it may appear that a model is evaluated on the wrong inputs (which are
        # then fixed by these hooks). Need a good way to present this to the
        # user.

        if not isinstance(doi_cut, InputCut):
            # Interventions only allowed onto one layer (see FIXME below.)

            # Define the hookfn.
            counter = 0

            def intervene_hookfn(self, inpt, outpt):
                nonlocal counter

                if input_timestep is None or input_timestep == counter:
                    # FIXME: generalize to multi-input layers. Currently can
                    #   only intervene on one layer.

                    # TODO: figure out how to check the case where intervention
                    # is on something that will never be executed. Would be good
                    # to give a user a warning in that case.

                    # TODO: figure out whether this is needed
                    inpt = inpt[0] if len(inpt) == 1 else inpt

                    ModelWrapper._nested_assign(
                        inpt if doi_cut.anchor == 'in' else outpt,
                        intervention.first_batchable(B)
                    )

                counter += 1

            # Register according to the anchor.
            if doi_cut.anchor == 'in':
                in_handle = (
                    self._get_layer(doi_cut.name).register_forward_pre_hook(
                        partial(intervene_hookfn, outpt=None)
                    )
                )
            else:
                if doi_cut.anchor is not None and doi_cut.anchor != 'out':
                    tru_logger.warning(
                        f'Unrecognized doi_cut.anchor {doi_cut.anchor}. Defaulting to `out` anchor.'
                    )
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
                inpt = om_of_many(inpt)

                if anchor == 'in':
                    hooks[layer_name] = inpt
                else:
                    # FIXME : will not work for multibranch outputs
                    # needed to ignore hidden states of RNNs
                    hooks[layer_name] = outpt

            return hookfn

        handles = [
            self._get_layer(name).register_forward_hook(
                get_hookfn(name, anchor)
            ) for name, anchor in names_and_anchors if name is not None
        ]

        with memory_suggestions(device=self.device):
            # Run the network.
            try:
                if self.force_eval:
                    self._model.eval()  # needed for determinism sometimes
                output = model_inputs.call_on(self._model)

                if isinstance(output, tuple):
                    output = output[0]

            finally:
                # Need to clean these up even if memory_suggestions catches the error.

                if not isinstance(doi_cut, InputCut):
                    # Clean up in handle.
                    in_handle.remove()

                # Clean up out handles.
                for handle in handles:
                    handle.remove()

        extract_args = dict(
            hooks=hooks, output=output, model_inputs=model_inputs
        )

        if attribution_cut:
            return (
                self._extract_outputs_from_hooks(cut=to_cut, **extract_args),
                self._extract_outputs_from_hooks(
                    cut=attribution_cut, **extract_args
                )
            )
        else:
            return (
                self._extract_outputs_from_hooks(cut=to_cut,
                                                 **extract_args), None
            )

    def _qoi_bprop(
        self, qoi: QoI, model_inputs: ModelInputs, doi_cut: Cut, to_cut: Cut,
        attribution_cut: Cut, intervention: TensorArgs
    ) -> Outputs[
            Inputs[TensorLike]
    ]:  # one outer element per QoI, one inner element per attribution_cut input

        B = get_backend()

        y, zs = self._fprop(
            model_inputs=model_inputs,
            doi_cut=doi_cut,
            to_cut=to_cut,
            attribution_cut=attribution_cut,
            intervention=intervention
        )

        def scalarize(t: torch.Tensor) -> torch.tensor:
            if len(t.shape) > 1 and np.array(t.shape).prod() != 1:
                # Adding warning here only if there is more than 1 dimension
                # being summed. If there is only 1 dim, its likely the batching
                # dimension so sum there is probably expected.
                tru_logger.warning(
                    f'Attribution tensor is not scalar (it is of shape {t.shape} '
                    f'and will be summed. This may not be your intention.'
                )

            return B.sum(t)

        y = to_cut.access_layer(y)
        zs = doi_cut.access_layer(zs)

        qois_out: Outputs[Tensor] = qoi._wrap_public_call(y)
        grads_list = [[] for _ in qois_out]

        for qoi_index, qoi_out in enumerate(qois_out):
            qoi_out: TensorLike = scalarize(qoi_out)

            try:
                with memory_suggestions(device=self.device):
                    grads_for_qoi = B.gradient(qoi_out, zs)

            except RuntimeError as e:
                if 'cudnn RNN backward can only be called in training mode' in str(
                        e):
                    raise RuntimeError(
                        "Cannot get deterministic gradients from RNN's with cudnn. See more about this issue here: https://github.com/pytorch/captum/issues/564 .\n"
                        "Consider setting 'torch.backends.cudnn.enabled = False' for now."
                    )
                raise e

            for grad_for_qoi in grads_for_qoi:
                grads_list[qoi_index].append(grad_for_qoi)

        del y  # TODO: garbage collection

        return grads_list

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
