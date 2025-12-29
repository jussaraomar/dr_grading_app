# config/params.py

class TrainingParams:
    #BATCH_SIZE = 32
    BATCH_SIZE_STAGE1 = 32
    BATCH_SIZE_STAGE2 = 16
    NUM_EPOCHS_STAGE1 = 10
    NUM_EPOCHS_STAGE2 = 15
    LEARNING_RATE_STAGE1 = 3e-4
    LEARNING_RATE_STAGE2 = 3e-5
    NUM_CLASSES = 5
    
class ModelParams:
    IMAGE_SIZE = (512, 512)
    VIT_IMAGE_SIZE = (224, 224)
    DROPOUT_RATE = 0.3