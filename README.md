# Use of reservoir computing to construct scar functions

Using Reservoir computing to study the quartic oscillator, a quantum chaotic system.

The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks. The whole code is the result of the work in <a href = "https://arxiv.org/abs/" target="_blank"> this paper</a>. Any contribution or idea to continue the lines of the proposed work will be very welcome.

**Remark**: We recommend the readers to view the notebooks locally and in *Trusted* mode for nicer presentation and correct visualization of the figures. 

In this work, a Reservoir Computing-based model is develop to propagate quantum wavefunctions with time.  In this project we obtain the eigenenergies, eigenstates and scar functions of this quantum system, by propagating an initial wavefunction with time using Reservoir Computing.

<p align="center"><img src="https://github.com/laiadc/Scars_RC/blob/main/example_scars.PNG"  align=middle width=600pt />
</p>
<p align="center">
<em>Example of the scar functions obtained in this work, for different energies. </em>
</p>

We compute the scar functions for four charasteristic periodic orbits: horizontal, quadruple-loop, square and triangle. Moreover, we compute the eigenenergies and eigenstates around energies E=1,10,100.

## Notebooks

All the notebooks used for this work can be found inside the folder **notebooks** .

**Remark**: Some of the training data could not be uploaded because it exceeded the maximum size allowed by GitHub. The notebooks provide the code to obtain such training data. 

### [RC Quartic Potential-E1.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-E1.ipynb)
Application of the adapted Reservoir Computing model to obtain the eigenenergies and eigenfunctions of the quartic oscillator around energy E=1.

### [RC Quartic Potential-E10.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-E10.ipynb)
Application of the adapted Reservoir Computing model to obtain the eigenenergies and eigenfunctions of the quartic oscillator around energy E=10.

### [RC Quartic Potential-E100.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-E100.ipynb)
Application of the adapted Reservoir Computing model to obtain the eigenenergies and eigenfunctions of the quartic oscillator around energy E=100.

### [RC Quartic Potential-scars_horizontal.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-scars_horizontal.ipynb)
Application of the adapted Reservoir Computing model to obtain the scar functions for the horizontal periodic orbit.

### [RC Quartic Potential-scars_quadruple-loop.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-scars_quadruple-loop.ipynb)
Application of the adapted Reservoir Computing model to obtain the scar functions for the quadruple-loop periodic orbit.

### [RC Quartic Potential-scars_square.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-scars_square.ipynb)
Application of the adapted Reservoir Computing model to obtain the scar functions for the square periodic orbit.

### [RC Quartic Potential-scars_triangle.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/RC%20Quartic%20Potential-scars_triangle.ipynb)
Application of the adapted Reservoir Computing model to obtain the scar functions for the triangle periodic orbit.

### [Results-scars.ipynb](https://github.com/laiadc/Scars_RC/blob/main/notebooks/Results-scars.ipynb)
This notebook summarizes the results of the paper and provides the figures provided in the paper.

### BibTex reference format for citation for the Code
```
@misc{RCScars,
title={Use of reservoir computing to construct scar functions},
url={https://github.com/laiadc/Scars_RC/},
note={Using Reservoir computing to study the quartic oscillator, a quantum chaotic system.},
author={L. Domingo and F. Borondo},
  year={2022}
}


