"""Communication-efficiency plugins applied to what a client uploads each round.

Three sparsification variants are provided so that the thesis can measure, rather
than assume, where the cost of sparsification actually comes from:

  WeightSparsification        keep the largest 10% of the *weights*, upload those
  DeltaSparsification         keep the largest 10% of the *update* (w_local minus
                              w_global), upload those on top of the weights the
                              server already has
  DeltaSparsification(EF)     as above, but whatever the mask discarded is carried
                              into the next round instead of being thrown away

The distinction matters. Under weight sparsification every client zeroes a
different 90% of the model, and FedAvg averages those zeros in, so a weight one
client dropped is pulled toward zero in the global model. That erodes the model
rather than compressing the update, and it is the likely reason aggressive ratios
collapse. Under delta sparsification the server already holds the previous global
weights, so an omitted coordinate simply means "no change from me this round".

On top of those lossy steps sits a lossless one. Which coordinates get sent is a
compression decision; how they are written on the wire is not, and the obvious
four-bytes-per-index encoding is wasteful enough that at K = 0.1 the indices are
half the payload. `payload_codecs` handles that layer and the choice of codec is
configured, not hardcoded. It changes the reported byte count only, never a
weight.

Naming: this file's ratio is a fraction of the model's 450,258 weights and is
about bandwidth. It is unrelated to `[tool.fueba.scoring] top_k_features`, which
is a count of the most deviant features used when scoring a window. The two were
routinely confused; they share no code and no units.
"""

import os
from abc import ABC, abstractmethod

import numpy as np

from federated_ueba import payload_codecs


def _topk_mask(array, ratio):
    """Zero everything except the `ratio` largest-magnitude entries of a tensor.

    Applied per tensor rather than over a global ranking, so a small tensor keeps
    at least one entry. Returns the sparsified copy and the number kept.
    """
    flat = array.flatten()
    k = min(max(1, int(len(flat) * ratio)), len(flat))
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    sparse = np.zeros_like(flat)
    sparse[idx] = flat[idx]
    return sparse.reshape(array.shape), k


class CommunicationPlugin(ABC):
    """Base class for communication efficiency plugins."""

    # True when this plugin leaves most coordinates at zero, so the payload only
    # has to carry the survivors plus an encoding of where they sit. The manager
    # uses this to decide whether to charge for a sparse or a dense payload.
    makes_sparse = False

    @abstractmethod
    def apply_on_client(self, parameters, reference=None):
        """Process parameters on the client before uploading.

        `reference` is the global model the server sent at the start of the
        round. Plugins that only look at the uploaded weights ignore it.
        """

    @abstractmethod
    def apply_on_server(self, parameters):
        """Process parameters on the server after aggregation."""

    def transport_payload(self, processed_params):
        """The arrays that actually cross the wire.

        Usually the processed parameters themselves. Delta sparsification is the
        exception: it hands aggregation a complete model but only uploads the
        sparse delta, so it overrides this.
        """
        return processed_params


class StandardPlugin(CommunicationPlugin):
    def apply_on_client(self, parameters, reference=None):
        return parameters

    def apply_on_server(self, parameters):
        return parameters


class WeightSparsificationPlugin(CommunicationPlugin):
    """Sparsify the weights themselves. The original behaviour.

    At ratio 0.1 this drops 405,248 of the 450,258 weights; at 0.05, 427,763.
    """

    makes_sparse = True

    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def apply_on_client(self, parameters, reference=None):
        return [_topk_mask(p, self.ratio)[0] for p in parameters]

    def apply_on_server(self, parameters):
        return parameters


class DeltaSparsificationPlugin(CommunicationPlugin):
    """Sparsify the update rather than the weights.

    The client uploads `reference + sparse(delta)`, so aggregation still receives
    a complete model and FedAvg is unchanged. Only the non-zero part of the delta
    has to cross the wire, because the server already holds `reference`; that is
    what `transport_payload` hands back for measurement.

    With `error_feedback`, coordinates the mask dropped are added back into the
    next round's delta instead of being discarded. A consistently small but
    non-zero update then eventually accumulates past the threshold and gets sent,
    rather than being suppressed forever. The residual is kept on disk because
    Flower rebuilds client objects every round, so in-memory state would not
    survive.
    """

    makes_sparse = True

    def __init__(self, ratio=0.1, error_feedback=False, residual_dir=None,
                 partition_id=None):
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.residual_dir = residual_dir
        self.partition_id = partition_id
        self._sparse_delta = None

    def _residual_path(self):
        if not (self.error_feedback and self.residual_dir and self.partition_id is not None):
            return None
        return os.path.join(self.residual_dir, f"residual_client_{self.partition_id}.npz")

    def _load_residual(self, shapes):
        path = self._residual_path()
        if path and os.path.exists(path):
            with np.load(path) as data:
                stored = [data[f"arr_{i}"] for i in range(len(data.files))]
            if [s.shape for s in stored] == shapes:
                return stored
        return [np.zeros(shape, dtype=np.float32) for shape in shapes]

    def _save_residual(self, residual):
        path = self._residual_path()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, *residual)

    def apply_on_client(self, parameters, reference=None):
        if reference is None:
            # This used to fall back to weight sparsification so the run would
            # still produce a result. That is the worst possible failure mode
            # here: the two variants differ in what they do to the global model,
            # not in how fast they are, and the fallback would quietly relabel a
            # delta experiment as a weight one. Measured on real checkpoints,
            # weight sparsification leaves 60% of the global model at exactly
            # zero at ratio 0.1; delta sparsification leaves none. A run that
            # silently switched between them mid-experiment would be
            # uninterpretable and would look like nothing worse than noise.
            raise RuntimeError(
                "Delta sparsification was asked for an update with no reference "
                "model. The client sets its reference in set_parameters, so this "
                "means get_parameters ran before the server sent a model. Fix "
                "the call order rather than sparsifying the weights instead: the "
                "two are different experiments.")

        deltas = [p.astype(np.float32) - r.astype(np.float32)
                  for p, r in zip(parameters, reference)]

        if self.error_feedback:
            residual = self._load_residual([d.shape for d in deltas])
            deltas = [d + res for d, res in zip(deltas, residual)]

        sparse_deltas = [_topk_mask(d, self.ratio)[0] for d in deltas]
        self._sparse_delta = sparse_deltas

        if self.error_feedback:
            self._save_residual([d - sd for d, sd in zip(deltas, sparse_deltas)])

        # Reconstruct a full model so aggregation stays plain FedAvg. Only the
        # sparse delta is charged for transport.
        return [r + sd for r, sd in zip(reference, sparse_deltas)]

    def apply_on_server(self, parameters):
        return parameters

    def transport_payload(self, processed_params):
        """The sparse delta, in whatever precision the payload ends up using.

        The delta is computed in float32; if quantization runs after this plugin
        the wire values are float16, so the delta is cast to match rather than
        being charged at a precision that was never sent.
        """
        if self._sparse_delta is None:
            return processed_params
        dtype = processed_params[0].dtype if processed_params else np.float32
        return [d.astype(dtype) for d in self._sparse_delta]


class QuantizationPlugin(CommunicationPlugin):
    def apply_on_client(self, parameters, reference=None):
        return [p.astype(np.float16) for p in parameters]

    def apply_on_server(self, parameters):
        return [p.astype(np.float32) for p in parameters]


class PluginManager:
    """Runs a chain of plugins and reports the resulting payload size.

    `codec` and `entropy` only affect the *reported* byte count, never the
    weights: they are lossless encodings of a payload whose contents are already
    decided. Both are applied to the dense baseline as well, so a saving reported
    against it comes from sparsification and quantization rather than from the
    entropy coder running on one side of the comparison only.
    """

    def __init__(self, plugins, codec="bitmask", entropy="none"):
        self.plugins = plugins
        self.codec = codec
        self.entropy = entropy

    def apply_on_client(self, parameters, reference=None):
        for plugin in self.plugins:
            parameters = plugin.apply_on_client(parameters, reference=reference)
        return parameters

    def apply_on_server(self, parameters):
        for plugin in reversed(self.plugins):
            parameters = plugin.apply_on_server(parameters)
        return parameters

    @property
    def sends_sparse_payload(self):
        return any(plugin.makes_sparse for plugin in self.plugins)

    def wire_payload(self, processed_params):
        """The arrays a client would actually upload, after every plugin ran."""
        for plugin in self.plugins:
            processed_params = plugin.transport_payload(processed_params)
        return processed_params

    def measure_transport_size(self, processed_params):
        """Bytes this upload occupies on the wire, obtained by encoding it."""
        payload = self.wire_payload(processed_params)
        if self.sends_sparse_payload:
            return payload_codecs.measure_payload(payload, self.codec, self.entropy)
        return payload_codecs.measure_dense_payload(payload, self.entropy)


class DownlinkCodec:
    """What the server does to the global model before broadcasting it.

    Measuring one baseline run showed the uplink is a minority of the traffic.
    Over 50 rounds the clients uploaded 2,003 MB and downloaded 6,441 MB, so
    compressing only what clients send caps the achievable saving at 24% of the
    total no matter how good the uplink compression gets. The downlink had been
    charged at raw dense fp32 while the uplink was charged through an entropy
    coder, which was also an inconsistency in the measurement rather than a
    property of the transport.

    Deliberately not the uplink `PluginManager`. The two directions differ in
    what they are even allowed to do:

      uplink    each client sends its own payload, so a client may sparsify
                against the global model it was given and the server can
                reconstruct the update from what it already holds
      downlink  the server sends one identical payload to every client, so
                anything client-specific stops it being a broadcast, and a delta
                against "the previous global model" is only decodable by a client
                that actually holds that model. At fraction_fit 0.5 a client
                misses half the rounds, and a client sampled for the first time
                holds nothing at all.

    That asymmetry is why only quantization and entropy coding are offered here.
    Both are stateless: the payload decodes from itself. Delta compression of the
    downlink would need a per-client synchronisation state and a full-model
    fallback for stale clients, which is a larger change than this one and was
    left out rather than half-built.

    Quantization here is lossy and genuinely changes the experiment: clients
    start each round from an fp16-rounded model rather than the fp32 average. It
    is measured, not assumed harmless.
    """

    def __init__(self, quantize, entropy):
        self.quantize = quantize
        self.entropy = entropy

    def compress(self, parameters):
        """Server side, before broadcast. Lossless steps do not appear here.

        Entropy coding is not applied to the arrays themselves because it happens
        at the transport layer: the bytes are compressed on the way out and
        decompressed on the way in, and the client sees the same values either
        way. It changes `measure` and nothing else.
        """
        if not self.quantize:
            return parameters
        return [p.astype(np.float16) for p in parameters]

    def decompress(self, parameters):
        """Client side, on receipt. Back to the precision the model trains in."""
        return [np.asarray(p, dtype=np.float32) for p in parameters]

    def measure(self, parameters):
        """Bytes this broadcast occupies, measured by encoding it.

        Called on what actually arrived, so if `compress` cast to fp16 this is
        charged at fp16. Dense rather than sparse: the downlink carries every
        coordinate, which is the whole reason it is the larger direction.
        """
        return payload_codecs.measure_dense_payload(parameters, self.entropy)


def get_downlink_codec(config):
    """Assemble the downlink codec from [tool.fueba.efficiency]."""
    modes = config.get("efficiency", "downlink_plugins")
    for mode in modes:
        if mode != "quantization":
            raise ValueError(
                f"Unknown downlink plugin {mode!r}. The downlink supports only "
                f"'quantization'; see DownlinkCodec for why sparsification "
                f"cannot be applied to a broadcast.")
    return DownlinkCodec(
        quantize="quantization" in modes,
        entropy=config.get("efficiency", "downlink_entropy_coder"),
    )


# Accepted spellings for each mode. "top_k" is the pre-rename name for weight
# sparsification and is kept so older experiment files still run.
_MODES = {
    "weight_sparsification": "weight",
    "sparsification": "weight",
    "top_k": "weight",
    "delta_sparsification": "delta",
    "delta_sparsification_ef": "delta_ef",
    "quantization": "quantization",
}


def _build_plugin(mode, ratio, residual_dir, partition_id):
    kind = _MODES.get(mode)
    if kind is None:
        raise ValueError(f"Unknown efficiency plugin {mode!r}. Expected one of "
                         f"{sorted(_MODES)}.")
    if kind == "weight":
        return WeightSparsificationPlugin(ratio=ratio)
    if kind == "delta":
        return DeltaSparsificationPlugin(ratio=ratio, error_feedback=False)
    if kind == "delta_ef":
        return DeltaSparsificationPlugin(
            ratio=ratio, error_feedback=True,
            residual_dir=residual_dir, partition_id=partition_id)
    return QuantizationPlugin()


def get_plugin(config, partition_id=None):
    """Assemble the plugin chain this experiment asked for.

    The ratio key is `sparsification_ratio`. It used to be `top_k_ratio`, which
    had to go because "top-k" also names the feature count in the scoring stage
    and the two were routinely confused. A configuration still using the old name
    is caught by the schema check at startup rather than here, since nothing in
    [tool.fueba.efficiency] may be a key the code does not know.
    """
    modes = config.get("efficiency", "active_plugins")
    ratio = config.get("efficiency", "sparsification_ratio")
    residual_dir = config.get("data", "scaler_dir")

    plugins = [_build_plugin(mode, ratio, residual_dir, partition_id)
               for mode in modes]
    return PluginManager(
        plugins,
        codec=config.get("efficiency", "payload_codec"),
        entropy=config.get("efficiency", "entropy_coder"),
    )
