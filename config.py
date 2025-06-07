# dir with ESC50 data
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100
n_mels = 128
hop_length = 512
#n_fft = 2048

#n_mfcc = 42

# model_constructor = "AudioMLP(n_steps=431,\
# n_mels=config.n_mels,\
# hidden1_size=512,\
# hidden2_size=128,\
# output_size=config.n_classes,\
# time_reduce=1)"

#model_constructor = "resnet.resnet18_audio(num_classes=config.n_classes)"
model_constructor = "resnet.resnet14_audio(num_classes=config.n_classes)"
#model_constructor = "resnet.resnet34_audio(num_classes=config.n_classes)"
#model_constructor = "cnn.cnn_small_audio(num_classes=config.n_classes)"
#model_constructor = "cnn.cnn_medium_audio(num_classes=config.n_classes)"
#model_constructor = "cnn.cnn_large_audio(num_classes=config.n_classes)"

# ###TRAINING
# ratio to split off from training data
val_size = .15  # could be changed
device_id = 0
batch_size = 48
# in Colab to avoid Warning
num_workers = 4
#num_workers = 0
# for local Windows or Linux machine
# num_workers = 6#16
persistent_workers = True
#persistent_workers = False
epochs = 120
#epochs = 1
# early stopping after epochs with no improvement
patience = 15
lr = 55e-5
weight_decay = 1e-4
beta1 = 0.9  # AdamW Beta1 Parameter
beta2 = 0.999  # AdamW Beta2 Parameter
eps = 1e-8   # AdamW Epsilon
warm_epochs = 15
dropout_rate = 0.3
gamma = 0.8
step_size = 3

lr_scheduler = "cosine_warm_restarts"  # Optionen: "step", "cosine", "cosine_warm_restarts"
lr_min_factor = 0.001  # Minimale LR = Ausgangs-LR * lr_min_factor
cosine_cycle_epochs = 15  # Für CosineAnnealingWarmRestarts
cosine_cycle_mult = 2  # Für CosineAnnealingWarmRestarts

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['best_val_loss.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2025-04-07-00-00'
test_experiment = 'results/sample-run '
