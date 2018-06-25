Classification-Breast-Cancer-Diagnosis

The data set is provided by Kaggle website:
https://www.kaggle.com/yuqing01/breast-cancer/data

In this project we are going to use 30 different columns to predict the Stage of Breast Cancer M (Malignant) and B (Benign)
This analysis has been done using Basic Machine Learning Algorithm with detailed explanation
Attribute Information:
    1. ID number
    2. Diagnosis (M = malignant, B = benign)
    3-32.Ten real-valued features are computed for each cell nucleus:
      a) radius (mean of distances from center to points on the perimeter)
      b) texture (standard deviation of gray-scale values)
      c) perimeter
      d) area
      e) smoothness (local variation in radius lengths)
      f) compactness (perimeter^2 / area - 1.0)
      g) concavity (severity of concave portions of the contour)
      h) concave points (number of concave portions of the contour)
      i) symmetry
      j) fractal dimension ("coastline approximation" - 1)

  here 3- 32 are divided into three parts first is Mean (3-13), Stranded Error(13-23) and Worst(23-32) and each contain 10 parameter
 (radius, texture,area, perimeter, smoothness,compactness,concavity,concave points,symmetry and fractal dimension)
  Here Mean means the means of the all cells, standard Error of all cell and worst means the worst cell
