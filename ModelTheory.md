# Preprocessing of data

# different experts (3, 4)

# For ech expert we are using efficient net b0 with pretrained weight of imagenet,

# We are doing fine tune of last few layers

# In the same way we have 3 different expert model (with imagenet)

# Lean everything about efficient net b0 and how it extract features from image and why it is better

# Output of each expert is dense layer with 46 classes


# create_enhanced_gating_network
    - in enhanced gating it is taking image input and doing convolution and according to the it determing for with expert to give how much weight, in output of that layer in dense in equal to number of experts 


Practice from this file
expertModelNumpy


# Findings

- number of experts, (efficientNet PreTrained)
- types of experts, (CNN Non Pretrained)
- Gating of experts ( Averaging,  Using CNN for weighted gating for experts)


when i use 3 experts, all efficientNet and Average Gating -> Model Did not converge 
    ( all pretrained model, number of trainable parameter are high arround 10 Million)

when i use 3 experts but 2 efficientNet and One Custom (Non Pretrained CNN) with Average Gating -> Model is Converged
    ( with arround 9 million trainable parameter)


I want to try what if 
    use 3 experts, all efficientNet but with CNN Gating -> 
   

what happens when back propagation when there are experts, same signal goes to all experts



# FeedBack
- create efficientNetB0 seperat model
- Custom CNN seperate
- On VGG Model train on that
- and apply score fusion





#  Model Strucutre

1. Create One EfficientNetB0 in pytorch -> keep it's metrics  - MODEL_CODE (efficientnet-b0-only)
2. Create One CNN Custom Model in keras -> keep it's metrics - MODEL_CODE (cnn-custom-only)
3. Create another model using Vgg16 Model in keras -> keep it's metrics - MODEL_CODE (vgg-16-only)


Mixture Of Expert Model:

1. Create Mixture of Expert model, use two model as expert (efficientNetB0) and (CustomCNN) and concate feature of these two model and predict the result -> keep it's metrics - MODEL_CODE (efficentnet_cnn_mixture_of_expert_concate_feature)

2. Create Another Mixture Of Expert model, use two model as expert (efficientNetB0) as (CustomCNN) and concate the score of these two model (making score fusion) and predict the result --> keep it's metrics - MODEL_CODE (efficentnet_cnn_mixture_of_expert_score_fusion)


3. Create Mixture of Expert model, use two model as expert (efficientNetB0) and (VGG16) and concate feature of these two model and predict the result -> keep it's metrics - MODEL_CODE (efficentnet_vgg16_mixture_of_expert_concate_feature)

4. Create Another Mixture Of Expert model, use two model as expert (efficientNetB0) as (CustomCNN) and concate the score of these two model (making score fusion) and predict the result --> keep it's metrics - MODEL_CODE (efficentnet_cnn_mixture_of_expert_score_fusion)


5. Create Mixture of Expert model, use two model as expert (CustomCNN) and (VGG16) and concate feature of these two model and predict the result -> keep it's metrics - MODEL_CODE (cnn_vgg16_mixture_of_expert_concate_feature)


6. Create Another Mixture Of Expert model, use two model as expert (CustomCNN) as (VGG16) and concate the score of these two model (making score fusion) and predict the result --> keep it's metrics - MODEL_CODE (cnn_vgg16_mixture_of_expert_score_fusion)




create a short cut for each model and in these different model write code to compare the results and prepare a table for them with traning and test.

User AdamW Optimizer.

It will take more time to generate all at one shot, prepare model one by one. i will ask you give me one model one by one.