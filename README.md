## Sources:

https://doi.org/10.1103/PhysRevE.106.014503 \
"Deep learning and inverse discovery of polymer self-consistent field theory inspired by physics-informed neural networks" \
Phys. Rev. E 106, 014503

https://doi.org/10.1016/j.jcp.2021.110519 \
"Deep learning and self-consistent field theory: A path towards accelerating polymer phase discovery" \
Journal of Computational Physics Volume 443, 15 October 2021, 110519

https://doi.org/10.1088/1367-2630/ab68fc \
"Phase diagrams of polymer-containing liquid mixtures with a theory-embedded neural network" \
Issei Nakamura 2020 New J. Phys. 22 015001

https://doi.org/10.1016/j.commatsci.2020.110224 \
"Deep learning model for predicting phase diagrams of block copolymers" \
Computational Materials Science Volume 188, 15 February 2021, 110224

## Notes:

# Deep learning model for predicting phase diagrams of block copolymers
 - Uses 3DCNN
 - Trained on 3000 metastable structures
 - Inputs were sets of volume fractions of each segment type in each 32 × 32 × 32 grid voxel (obtained from SCFT calculation) and...
 - a one-hot vector of the five types of stable phase labels for the corresponding set of polymer structures and χN values
 - Layers = four convolutional layers, two max-pooling layers, a flatten layer, 2 dense layers

   ![image](https://github.com/user-attachments/assets/c0a21083-7531-404f-a6a5-a970dedc99c3)

