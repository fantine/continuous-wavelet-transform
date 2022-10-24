# Two-Dimensional Continuous Wavelet Transform

- **Author:** Fantine Huot

This is a library for two-dimensional continuous wavelet transforms (2D CWT) for seismic processing.

2D CWT filtering addresses the challenges arising from temporally and spatially transient phases,
which cannot be effectively addressed using conventional stationary filtering. 
Using the 2D CWT we can transform seismic data into a domain in which the signal of interest and other wave modes are well-separated.
It is then possible to characterize the noise in this domain and design a filter to remove the unwanted noise modes.
The data can then be transformed back to the original time domain. 
The described method is computationally robust and its theory can be extended to higher dimensions.
As a consequence, the methodology is applicable to any temporally and spatially continuous seismic dataset, both pre-stack and after imaging.  

# Citation 

If using this code in a publication, presentation, or other research product please use the following citation:

[Fantine Huot, Biondo Biondi, Anthony Lichnewsky, and Carlos Boneti, (2019), "Automatic denoising by 2-D continuous wavelet transform," SEG Technical Program Expanded Abstracts : 3944-3948.](https://library.seg.org/doi/abs/10.1190/segam2019-3213958.1)

[https://doi.org/10.1190/segam2019-3213958.1](https://doi.org/10.1190/segam2019-3213958.1)
