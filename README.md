# The troublesome kernel: Why deep learning for inverse problems is typically unstable

This repository contains code related to the paper "The troublesome kernel: Why
deep learning for inverse problems is typically unstable", by N. Gottschling,
V. Antun, B. Adcock and A. C. Hansen.  

This repository does only contains the source code. The data (3.7 GB) can be downloaded
here
[https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage3.zip](https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage3.zip).
Note that you will most likely have to change all paths so that they point to the 
correct data, in order for the scripts to run smoothly. In most cases, this can be done in the configuration files.

Note that in order to run the
[DeepMRINet](https://github.com/js3611/Deep-MRI-Reconstruction) and 
[FBPConvNet](https://github.com/panakino/FBPConvNet), you will have to download and install the code related to the networks.

For the sparse regularization reconstructions we have used the
[ShearletReweighting](https://github.com/jky-ma/ShearletReweighting)
code from J. Ma & M. März paper and
[spgl1](https://github.com/mpf/spgl1).  These repositories must also
be downloaded and added to your Matlab path.


