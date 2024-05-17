import json
# Data to write
config = {
"batchSize": 32,
"epochs": 160,
"scheduled": True,
"lr": 0.001,
"inputDim": (16,16), # Input dim of the encoded
"numClass": 256, # Num classes = possible pixel values
"channels": 128, # Num channels intermediate feature representation
"kernel": 5, # Kernel size
"blocks": 8,
"resBlocks":8,
"resChannels": 128,
"attention": True,
"dropout": 0.4,
"condResChannels": 32, # Number of channels in the conditional ResNet
"condResKernel": 3, # Size of the kernel in the conditional ResNet
"condResBlocks":2, # Number of conditional residual blocks in the conditional ResNet
"outResBlock": 4 # Number of residual blocks in the output layer
}
# Write to file
with open('transistor.json', 'w') as json_file:
    json.dump(config, json_file, indent=4)