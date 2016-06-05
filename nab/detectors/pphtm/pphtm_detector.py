
import os
import math
import simplejson as json

from pphtm.pphtm.pphtm_brain import PPHTMBrain
from nupic.encoders.scalar import ScalarEncoder
from nab.detectors.base import AnomalyDetector

PPHTM_CONFIG = {
    # 'PROXIMAL_ACTIVATION_THRESHHOLD': 3,
    # 'DISTAL_ACTIVATION_THRESHOLD': 2,
    # 'BOOST_MULTIPLIER': (1.0, 2.9), # 2.58
    # 'DESIRED_LOCAL_ACTIVITY': 2,
    # 'DISTAL_SYNAPSE_CHANCE': 0.5,
    # 'TOPDOWN_SYNAPSE_CHANCE': 0.5,
    # 'MAX_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.6,
    # 'MIN_PROXIMAL_INIT_SYNAPSE_CHANCE': 0.1,
    'CELLS_PER_REGION': 8**2,
    'N_REGIONS': 1,
    # 'BIAS_WEIGHT': (0.5, 0.9),
    # 'OVERLAP_WEIGHT': (0.4, 0.6),
    # 'FADE_RATE': (0.2, 0.6),
    'DISTAL_SEGMENTS': 3,
    # 'PROX_SEGMENTS': 2,
    # 'TOPDOWN_SEGMENTS': 2, # Only relevant if >1 region
    # 'SYNAPSE_DECAY_PROX': 0.0008,
    # 'SYNAPSE_DECAY_DIST': 0.0008,
    # 'PERM_LEARN_INC': 0.08,
    # 'PERM_LEARN_DEC': 0.05,
    'CHANCE_OF_INHIBITORY': 0.1,
    # 'SYNAPSE_ACTIVATION_LEARN_THRESHHOLD': 1.0,
    # 'DISTAL_BOOST_MULT': 0.02,
    # 'INHIBITION_RADIUS_DISCOUNT': 0.8,
    # Booleans
}


class PphtmDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):

        super(PphtmDetector, self).__init__(*args, **kwargs)

        self.b = None # Brain
        self.encoder = None

    def getAdditionalHeaders(self):
        return ["totalBias", "predictedBias"]

    def handleRecord(self, inputData):
        value = inputData.get('value')
        inputs = self.encode_data(value)
        self.b.process(inputs, learning=True)
        anomalyScore, totalBias, predictedBias = self.b.get_anomaly_score()
        return (anomalyScore, totalBias, predictedBias)

    def encode_data(self, data):
        return self.encoder.encode(data)

    def initialize(self):
        N_INPUTS = 10**2
        self.b = PPHTMBrain(min_overlap=1, r1_inputs=N_INPUTS)
        self.b.initialize(**PPHTM_CONFIG)

        rangePadding = max([abs(self.inputMax - self.inputMin) * 0.2, .01])
        minVal=self.inputMin-rangePadding
        maxVal=self.inputMax+rangePadding

        self.encoder = ScalarEncoder(n=N_INPUTS, w=5, minval=minVal, maxval=maxVal, periodic=False, forced=True)

