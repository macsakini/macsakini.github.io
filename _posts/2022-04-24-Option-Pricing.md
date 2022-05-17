---
classes: wide
category: Financial Modelling
---

Option Analysis and Prediction using Fast Fourier Transform and Levy Alpha Stable Distributions

This project uses a number of concepts.

- Black Scholes model.

- Capital Assets Pricing Model (CAPM).

- Levy Alpha Stable Distributions. 

- Market Model.

- Fourier and Inverse Fourier Transformations.

The files of the project are all over the place but each file deals with a certain topic. I have also included a docx file explaining the entire theorem in the data folder. 

However the following concept can be critical to understand.
1. Options are regularly proced using the BS model.

2. Volatility in stock markets and securities are regularly functions of normal distributions according to a number of models. 

![image](https://user-images.githubusercontent.com/47692036/161438930-7d658922-079a-4a16-8634-f748ea50e9df.png)
Figure 1.1 CAPM Model

3. Technical analysis incorporates normal distributions. This can also be seen in the thrid graph in the image above. The residuals are trying to get fit under a normal curve.

4. Beta Coefficient explains risk in a regression model of market prices with underlying asset.

5. Beta distributions can take different forms depending on the shape and scale parameters assigned to it. **Bet coefficients are not related to beta distributions**

6. Prior analysis has shown that in some cases fitting a different distribution similar to the normal explains better the fractals and risk numbers in a market.

7. Levy Distributions are extracts of the stable distributions. According to wikipedia, beta distributions are not __""not analytically expressible, except for some parameter values""__. Normal distribution is also a stable distribution with an alpha = 2

8. This being so, - All stable distributions are infinitely divisible. - With the exception of the normal distribution (α = 2), stable distributions are leptokurtotic and heavy-tailed distributions. 

9. The characteristic function φ(t) of any probability distribution is just the __Fourier transform__ of its probability density function f(x). 

10. The __density function (pdf)__ is therefore the __inverse Fourier transform__ or the __fast fourier__ of the characteristic function.

__Although the probability density function for a general stable distribution cannot be written analytically, the general characteristic function can be expressed analytically.__

Having understoodm these ten simple facts. You already have a mind map of the whole paper. 

Instead of using normal use levy, but since levy does not have pdf, use fast fourier to find it. Fit the residuals of the market using the levy. Find the shape and scale parameters. Also the alpha and beta.

![image](https://user-images.githubusercontent.com/47692036/161438872-1ab6ac8a-5684-4f11-801c-9321dfdf7de9.png)
Figure 1.2 Dow Jones Levy Alpha Distribution

# The files
 - DJIA - Dow Jones
 - PF - Pfizer
 - fft - Fast Fourier Transform
djia.rmd is an R file making analysis on the Down Jones index for stated period.
pfe.rmd - analysis on pfizer stock for stated period.
fft.py - is the fast fourier transform implemented.
levystable.py -  is the levy stable distribution inside __scipy__ library.
hs.csv is a combined data file having side by side stocks of dow jones and pfizer.
pfe.csv has only pfizer historical prices

![image](https://user-images.githubusercontent.com/47692036/161438964-376587d3-d0e3-460f-815f-174ec2993158.png)
Figure 1.3 Predictions made using regular BS model.

![image](https://user-images.githubusercontent.com/47692036/161438984-9dd0e583-3b5d-4d0c-b91f-3a301b55fd8f.png)
Figure 1.4 Predictions made using levy alpha. 

