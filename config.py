# AER System Configuration

# Flask Settings
DEBUG = True
SECRET_KEY = 'myflasksecret123'

# Model Settings
MODEL_PATH = 'models/longformer_best_model.pkl'
MAX_LENGTH = 512
BATCH_SIZE = 8

# Training Settings
LEARNING_RATE = 2e-5
EPOCHS = 4
DROPOUT = 0.3

# API Settings
API_RATE_LIMIT = 100  # requests per hour
