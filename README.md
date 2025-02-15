The official code of "Shihao Zhang, Yuguang Yan, Angela Yao. Improving Deep Regression with Tightness. ICLR 2025."[[PDF]](https://openreview.net/pdf?id=dkoiAGjZV9)

## Visualization of the Feature manifold Updating 
<img src="feature_space_updating.gif" width="800">  
Throughout training, the representation (node) typically moves either toward or away from the direction of $\theta$ (i.e., the regressor). As a result, regression exhibits limited ability to tighten/compress representations in directions perpendicular to $\theta$. The movement can be regarded as a probability density shift. Visualization experiments are conducted on the NYU Depth V2 dataset for depth estimation, using 1000 fixed pixel-wise samples from a batch of 32 images. The feature dimension is set to 2 for visualization. At the beginning of training (i.e., iteration 0 to 100), the feature manifold undergoes significant scaling to align with the scale of the target and the regressor.


## Experiments on the real-world and synthetic datasets

You can use ICLR25_OT.py in the same way as OrdinalEntropy.py or PH_Reg.py from [OrdinalEntropy](https://github.com/needylove/OrdinalEntropy)/ [Ph-Reg](https://github.com/needylove/PH-Reg). Please refer to the instructions provided in the OrdinalEntropy or PH-Reg for detailed experimental settings.


