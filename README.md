# EMSTDP OOD
The Repo contains OOD detection website using EMSTDP python implementation.
### To get the outlier exposure weights of EMSTDP.
```bash
python EMSTDP_main.py
```
EMSTDP OOD adopts outlier exposure algorithm to train with 2 classes MSTAR data. EMSTDP uses MNIST as the outlier data. The file also generates AUROC curve and prints out FPR95, AUPR , AUROC results under the *get_and_print_results()* function.
Output EMSTDP weights:
- oe_w_h.npy: hidden layer weights after training with outlier samples
- oe_w_o.npy: output layer weights after training with outlier samples
### To run the website 
```bash
python app.py
```
The file contains Flask website implementation of OOD detection using pretrained EMSTDP outlier weights oe_w_h.npy and oe_w_o.npy.
### Requirements
-Flask 2.3.2
-Python >= 3.4
-SKlearn >= 1.2.0
-Numpy >= 1.24.0
-Pytorch >= 1.13.0
-Torch >= 0.13.0

