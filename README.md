# 828P2_Autoencoder

alex_pipeline.py is used to extract signatures from mutation ||| catalog usage: python alex_pipeline.py input_file.npy signature_file.npy number_of_signatures


denoising_autoencode.py is used to train the autoencoder ||| usage: python denoising_autoencoder.py input_file.npy (the results will be saved in the same folder, which includes the model trained, and two figures plotting the loss over time in training)


model_extract.py is used to put data through the denoising_autoencoder and save the result locally ||| usage: python model_extract.py model_file data_file number_of_hidden_state output_file_name
