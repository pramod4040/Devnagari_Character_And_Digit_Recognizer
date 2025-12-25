import numpy as np


# 1. Stack Expert model


# Simulate 3 experts' outputs, each of shape (2 samples, 2 classes)
expert1 = np.array([[1, 2], [3, 4]])
expert2 = np.array([[5, 6], [7, 8]])
expert3 = np.array([[9, 10], [11, 12]])

experts = [expert1, expert2, expert3]

# TODO: Stack the experts to get a single array of shape (2, 2, 3)
# Each (sample, class) should now have all 3 experts’ predictions stacked

stacked = np.stack(experts, axis=2)
print("Stacked shape:", stacked.shape)
print(stacked)



# Exercise 2: Simulate gate outputs and reshape

# Simulate gate outputs for 2 samples and 3 experts
gates = np.array([
    [0.2, 0.3, 0.5],  # sample 1
    [0.6, 0.1, 0.3]   # sample 2
])

# TODO: Reshape gate to broadcast with stacked expert outputs
# Final shape should be (2, 1, 3)
reshaped_gates = gates.reshape(2, 1, 3)
print("Reshaped gates shape:", reshaped_gates.shape)
print(reshaped_gates)


# Exercise 3: Weighted multiplication

# Use values from Exercise 1 & 2
# TODO: Multiply gates and expert predictions
weighted_outputs = reshaped_gates * stacked
print("Weighted outputs shape:", weighted_outputs.shape)
print(weighted_outputs)



# TODO: Sum across expert axis (axis=2)
final_output = np.sum(weighted_outputs, axis=2)
print("Final output shape:", final_output.shape)
print(final_output)


