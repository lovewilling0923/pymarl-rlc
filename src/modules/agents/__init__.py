REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .transformer import TransformerAggregationAgent
REGISTRY['transformer'] = TransformerAggregationAgent

from .central_rnn_agent import CentralRNNAgent
REGISTRY['central_rnn'] = CentralRNNAgent

from .rtc_agent import HistoryRTCs
REGISTRY["rtcs"] = HistoryRTCs
