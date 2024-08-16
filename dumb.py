import torch
import time

# make a dumb program that occupies the GPU for 50 seconds
print("Starting to occupy GPU")

# Create a tensor and move it to the GPU
tensor = torch.rand(10000, 10000).cuda()
time.sleep(30)

print("Finished occupying GPU")


