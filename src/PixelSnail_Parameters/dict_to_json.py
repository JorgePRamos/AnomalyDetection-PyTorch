import json
# Data to write
config = {
"batchSize": 64,
"epochs": 100,
"scheduled": True,
"lr": 0.001,
"inputDim": (16,16), # Input dim of the encoded
"numClass": 256, # Num classes = possible pixel values
"channels": 64, # Num channels intermediate feature representation
"kernel": 5, # Kernel size
"blocks": 4,
"resBlocks":4,
"resChannels": 64,
"attention": True,
"dropout": 0.3,
"condResChannels": 64, # Number of channels in the conditional ResNet
"condResKernel": 3, # Size of the kernel in the conditional ResNet
"condResBlocks":2, # Number of conditional residual blocks in the conditional ResNet
"outResBlock": 4 # Number of residual blocks in the output layer
}
# Write to file
with open('capsule.json', 'w') as json_file:
    json.dump(config, json_file, indent=4)