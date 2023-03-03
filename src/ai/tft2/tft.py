from ai.tft2.embedding import GenericEmbedding
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

d_model = 10

input_spec = {
    'static': {
        'day_of_week': {
            'num_categories': 5,
            'input_tensor': layers.Input(shape=())
        }
    },
    'observed': {
        'time_of_day_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        },
        'volume_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        },
        'open_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        },
        'high_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        },
        'low_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        },
        'close_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        },
        'vwap_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        }
    },
    'forecast': {
        'time_of_day_ts': {
            'num_categories': 0,
            'input_tensor': layers.Input(shape=())
        }
    }
}

for input_type, inputs in input_spec.items():
    for input_name, input_data in inputs.items():
        input_spec[input_type][input_name]['embedding_tensor'] = \
            GenericEmbedding(input_data['num_categories'], d_model)(input_data['input_tensor'])



target_spec = {
    'high_ts': {
        'num_categories': 0
    },
    'low_ts': {
        'num_categories': 0
    }
}
