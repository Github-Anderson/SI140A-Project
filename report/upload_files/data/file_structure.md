Introduction on files in this folder:

checkpoint0.ckpt: The checkpoint of the diffusion model. This checkpoint will be updated once you run the Jupyter Notebook.

raw_data_1.npy: The numpy array form of the file data1.csv, which is generated by python program that transforms csv to npy

raw_data_2.npy: The numpy array form of the file data2.csv, which is generated by python program that transforms csv to npy

raw_data_3.npy: The numpy array form of the file data3.csv, which is generated by python program that transforms csv to npy

simulated_data_from_raw_1.npy: The inference result of the diffusion model. Once you run the Jupyter Notebook on your computer, this file will be generated. This numpy array is $(1000, 3)$, standing for 1000 data with three people and 5 yuan in total. This file will be updated once you run the Jupyter Notebook.

simulated_data_from_raw_1_not_ipynb.npy: The inference result of our experiment. This numpy array is $(1000, 3)$, standing for 1000 data with three people and 5 yuan in total. We get this result from running the Jupyter Book on our computer. 

simulated_data_from_raw_2_not_ipynb.npy:  The inference result of our experiment. This numpy array is $(1000, 4)$, standing for 1000 data with four people and 5 yuan in total. We get this result from running the Jupyter Book on our computer. 

Note that if you want to generate 'simulated_data_from_raw_2.npy', you shall change the data file path in the Jupyter Notebook to 'data/raw_data_3.npy' and the output path. 









