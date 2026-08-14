run_exp.py calls the train.py and eval.py modules from the experiments folder for one or multiple experiment IDs, the setup of which needs to be defined in experiments/configs/config_{ID}.py

paths.json references "data" and "trained_models" directories (currently one directory level up from this project). Features for style transfer and corrupted data labels are also referenced here.

"trained_models" and the "results"-directory use an internal folder structure: results/{'datasetname'}/{'modelname'} and trained_models/{'datasetname'}/{'modelname'}.

Some model architectures in /experiments/models feature a parameter "factor". Such models use a stride=factor in the first convolution, e.g. to handle TinyImageNets 64px images with the same architecture as CIFARs 32px images, when no adaptive average pooling is used (standard WRN models). All models inherit forward pass methods from ct_model.py to handle normalization, noise injections, feature mixup, DeepAugment, MoEx within the forward pass.