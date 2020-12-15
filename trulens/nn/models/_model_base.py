""" 
The TruLens library is designed to support models implemented via a variety of
different popular python neural network frameworks: Keras (with TensorFlow or 
Theano backend), TensorFlow, and Pytorch. In order provide the same 
functionality to models made with frameworks that implement things (e.g., 
gradient computations) a number of different ways, we provide an adapter class 
to provide a unified model API. In order to compute attributions for a model, 
it should be wrapped as a `ModelWrapper` instance.
"""
from abc import ABC as AbstractBaseClass
from abc import abstractmethod


DATA_CONTAINER_TYPE = (list, tuple)


class ModelWrapper(AbstractBaseClass):
    """
    A wrapper interface for models that exposes the components needed for 
    computing attributions. This is intended to produce a consistent 
    functionality for all models regardless of the backend/library the model is 
    implemented with.
    """

    @abstractmethod
    def __init__(
            self,
            model,
            logit_layer=None,
            replace_softmax=False,
            softmax_layer=-1,
            custom_objects=None,
            input_shape=None,
            device=None,
            input_tensors=None,
            output_tensors=None,
            internal_tensor_dict=None,
            default_feed_dict=None,
            session=None):
        """
        Parameters:
            model:
                The model to wrap. For the TensorFlow 1 backend, this is 
                expected to be a graph object.

            logit_layer:
                Specifies the name or index of the layer that produces the
                logit predictions. Supported for Keras and Pytorch models.

            replace_softmax:
                _Supported for Keras models only._ If true, the activation
                function in the softmax layer (specified by `softmax_layer`) 
                will be changed to a `'linear'` activation. 

            softmax_layer:
                _Supported for Keras models only._ Specifies the layer that
                performs the softmax. This layer should have an `activation`
                attribute. Only used when `replace_softmax` is true.

            custom_objects:
                _Optional, for use with Keras models only._ A dictionary of
                custom objects used by the Keras model.

            input_shape:
                _Required for use with Pytorch models only._ Tuple specifying
                the input shape (excluding the batch dimension) expected by the
                model.

            device:
                _Optional, for use with Pytorch models only._ A string
                specifying the device to run the model on.

            input_tensors:
                _Required for use with TensorFlow 1 graph models only._ A list
                of tensors representing the input to the model graph.

            output_tensors:
                _Required for use with TensorFlow 1 graph models only._ A list
                of tensors representing he output to the model graph.

            internal_tensor_dict:
                _Optional, for use with TensorFlow 1 graph models only._ A
                dictionary mapping user-selected layer names to the internal
                tensors in the model graph that the user would like to expose.
                This is provided to give more human-readable names to the layers
                if desired. Internal tensors can also be accessed via the name
                given to them by tensorflow.

            default_feed_dict:
                _Optional, for use with TensorFlow 1 graph models only._ A
                dictionary of default values to give to tensors in the model
                graph.

            session:
                _Optional, for use with TensorFlow 1 graph models only._ A 
                `tf.Session` object to run the model graph in. If `None`, a new
                temporary session will be generated every time the model is run.
        """
        self._model = model

    @property
    def model(self):
        """
        The model this object wraps.
        """
        return self._model

    def __call__(self, x):
        """
        Shorthand for obtaining the model's output on the given input.

        Parameters:
            x:
                Input point.

        Returns:
            Model's output on the input point.
        """
        return self.fprop(x)[0]

    @abstractmethod
    def fprop(
            self,
            model_args,
            model_kwargs={},
            doi_cut=None,
            to_cut=None,
            attribution_cut=None,
            intervention=None,
            **kwargs):
        """
        **_Used internally by `AttributionMethod`._**

        Forward propagate the model beginning at `doi_cut`, with input 
        `intervention`, and ending at `to_cut`.

        Parameters:
            model_args, model_kwargs: 
                The args and kwargs given to the call method of a model.
                This should represent the instances to obtain attributions for,
                assumed to be a *batched* input. if `self.model` supports
                evaluation on *data tensors*, the  appropriate tensor type may
                be used (e.g., Pytorch models may accept Pytorch tensors in 
                addition to `np.ndarray`s). The shape of the inputs must match
                the input shape of `self.model`. 

            doi_cut:
                Cut defining where the Distribution of Interest is applied. The
                shape of `intervention` must match the input shape of the 
                layer(s) specified by the cut. If `doi_cut` is `None`, the input
                to the model will be used (i.e., `InputCut()`).

            to_cut:
                Cut defining the layer(s) at which the propagation will end. The
                If `to_cut` is `None`, the output of the model will be used
                (i.e., `OutputCut()`).
            
            attribution_cut:
                Cut defining where the attributions are collected. If 
                `attribution_cut` is `None`, it will be assumed to be the
                `doi_cut`
            
            intervention:
                The intervention created from the Distribution of Interest. If 
                `intervention` is `None`, then it is equivalent to the point
                DoI.

        Returns:
            (list of backend.Tensor or np.ndarray)
                A list of output activations are returned, keeping the same type
                as the input. If `attribution_cut` is supplied, also return the
                cut activations.
        """
        raise NotImplementedError

    @abstractmethod
    def qoi_bprop(
            self,
            qoi,
            model_args,
            model_kwargs={},
            doi_cut=None,
            to_cut=None,
            attribution_cut=None,
            intervention=None,
            **kwargs):
        """
        **_Used internally by `AttributionMethod`._**
        
        Runs the model beginning at `doi_cut` on input `intervention`, and 
        returns the gradients calculated from `to_cut` with respect to 
        `attribution_cut` of the quantity of interest.

        Parameters:
            model_args, model_kwargs: 
                The args and kwargs given to the call method of a model.
                This should represent the instances to obtain attributions for, 
                assumed to be a *batched* input. if `self.model` supports
                evaluation on *data tensors*, the  appropriate tensor type may
                be used (e.g., Pytorch models may accept Pytorch tensors in 
                addition to `np.ndarray`s). The shape of the inputs must match
                the input shape of `self.model`. 

            doi_cut:
                Cut defining where the Distribution of Interest is applied. The
                shape of `intervention` must match the input shape of the 
                layer(s) specified by the cut. If `doi_cut` is `None`, the input
                to the model will be used (i.e., `InputCut()`).

            to_cut:
                Cut defining the layer(s) at which the propagation will end. The
                If `to_cut` is `None`, the output of the model will be used
                (i.e., `OutputCut()`).
            
            attribution_cut:
                Cut defining where the attributions are collected. If 
                `attribution_cut` is `None`, it will be assumed to be the
                `doi_cut`
            
            intervention:
                The intervention created from the Distribution of Interest. If 
                `intervention` is `None`, then it is equivalent to the point 
                DoI.


        Returns:
            (backend.Tensor or np.ndarray)
                the gradients of `qoi` w.r.t. `attribution_cut`, keeping same
                type as the input.
        """
        raise NotImplementedError

    @staticmethod
    def _nested_assign(x, y):
        """
        _nested_assign Assigns tensors values in y to tensors in x.

        Parameters
        ----------
        x:  backend.Tensor or a nested list or tuple of backend.Tensor
            The leaf Tensors will be assigned values from y.
        y:  backend.Tensor or a nested list or tuple of backend.Tensor
            Must be of the same structure as x. Contains objects that
            will be assigned to x.
        """
        if isinstance(y, tuple) or isinstance(y, list):
            for i in range(len(y)):
                ModelWrapper._nested_assign(x[i], y[i])
        else:
            x[:] = y[:]

    @staticmethod
    def _nested_apply(y, fn):
        """
        _nested_apply Applies fn to tensors in y.

        Parameters
        ----------
        y:  non-collective object or a nested list/tuple of objects
            The leaf objects will be inputs to fn.
        fn: function
            The function applied to leaf objects in y. Should
            take in a single non-collective object and return
            a non-collective object.
        Returns
        ------
        non-collective object or a nested list or tuple
            Has the same structure as y, and contains the results
            of fn applied to leaf objects in y.

        """
        if isinstance(y, tuple) or isinstance(y, list):
            out = []
            for i in range(len(y)):
                out.append(ModelWrapper._nested_apply(y[i], fn))
            return tuple(out) if isinstance(y, tuple) else out
        else:
            return fn(y)

    @staticmethod
    def _flatten(x):
        """
        _flatten Given a nested list or tuple x, outputs a new
            flattened list.

        Parameters
        ----------
        x:  non-collective object or a nested list/tuple of objects
            The nested list to be flattened.
        Returns
        ------
        list
            New list containing the leaf objects in x.

        """
        if isinstance(x, list) or isinstance(x, tuple):
            out = []
            for i in range(len(x)):
                out.extend(ModelWrapper._flatten(x[i]))
            return out
        else:
            return [x]

    @staticmethod
    def _unflatten(x, z, count=None):
        """
        _unflatten Given a non-nested list x, outputs a new nested list
            or tuple of the same structure as z.

        Parameters
        ----------
        x:  list of non-collective objects.
            Contains the leaf objects of the nested list.
        z:  non-collective object or a nested list/tuple of objects
            Contains the structure that x will be unflattened to.
        Returns
        ------
        nested list/tuple
            New nested list/tuple containing the leaf objects in x
            and the same structure as z.

        """
        if not count:
            count = [0]
        if isinstance(z, list) or isinstance(z, tuple):
            out = []
            for i in range(len(z)):
                out.append(ModelWrapper._unflatten(x, z[i], count))
            return tuple(out) if isinstance(z, tuple) else out
        else:
            out = x[count[0]]
            count[0] += 1
            return out
