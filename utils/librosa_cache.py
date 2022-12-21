import os

## This must be called before anything else.
def enable_cache():
    os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'

import librosa 
def purge_cache():
    librosa.cache.clear()