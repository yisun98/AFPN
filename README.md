# AFPN (updating...)
Adaptive Detail Injection-Based Feature Pyramid Network For Pan-sharpening



### Fig.1 : the overall framework of AFPN.

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig1.png) 

**Fig.1 caption**:  Green and orange layers are spectral and spatial feature maps of spectral and spatial pyramids respectively. Resblock layer contains two convolutional layers, a PReLu activation function, and a residual connection. Upsample layer is Bicubic interpolation and Resample layer is a convolutional layer with stride=2.

### TI vs. FDI on AFPN

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig-tivsfdi.png) 

### FDI on GF-1

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig-fdigf1.png)

### More experiments

#### Texture loss

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig-tl.png)

#### Inference Time and Params

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig-time-params.png) 

### ADI on BPN

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig-bpn.png) 

### Mistakes in Paper

![image](https://github.com/yisun98/AFPN/blob/main/experiments/fig1-result.png)


