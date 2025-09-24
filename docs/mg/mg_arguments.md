# Glossary

This document provides definitions and explanations for the main arguments and variables used in Machine Gnostics data analytics, machine learning and deep learning models. Understanding these concepts will help users grasp the unique characteristics of the Machine Gnostics library, which is based on the non-statistical paradigm of Mathematical Gnostics.

---

## Core Concepts

- **Machine Gnostics**  
  A machine learning and deep learning library founded on Mathematical Gnostics, a non-statistical paradigm for data analysis.

- **Mathematical Gnostics**  
  An alternative to traditional statistical methods, focusing on the quantification and estimation of uncertainty in data.

---

## Key Arguments and Gnostic Characteristics

### 1. Gnostic Geometries

These are the fundamental variables used to describe data in the gnostic framework. They are divided into two main spaces:

- **Quantifying Space (Q-space, j):**  
  Describes the variability and irrelevance in the data.
- **Estimating Space (E-space, i):**  
  Describes the estimation of variability and relevance.

#### Quantifying Geometry

- **fj**: *Quantifying data variability*  
  Measures the variability present in the data.

- **hj**: *Quantifying irrelevance*  
  Measures the irrelevance or error due to variability.

#### Estimating Geometry

- **fi**: *Estimating data variability*  
  Provides an estimation of the data's variability.

- **hi**: *Estimating relevance*  
  Provides an estimation of the data's relevance.


All four variables (\( f_j, h_j, f_i, h_i \)) are called **gnostic characteristics**.

---

### 2. Probability Arguments

- **pi**: *Estimating probability*  
  Probability estimate in the context of the gnostic model.

- **pj**: *Quantifying probability*  
  Quantifies the probability based on the quantifying characteristics.

---

### 3. Information

- **Ii**: *Estimating information*  
  Information estimate for the data.

- **Ij**: *Quantifying information*  
  Quantifies the information content.


---

### 4. Entropy

- **ei**: *Estimating entropy*  
  Entropy estimate for the data.

- **ej**: *Quantifying entropy*  
  Quantifies the entropy content.

- **re**: *Residual entropy*  
  The remaining entropy after estimation, representing the difference between quantification and estimation entropy.

---

### 5. Loss Functions

- **Hc loss**: *Gnostic mean relevance loss*  
  A loss function based on gnostic relevance, where \( c \) can be \( i \) or \( j \).

---



## Further reading

For more detailed mathematical background, see the foundational texts on [Mathematical Gnostics](../ref/references.md) and the documentation of the Machine Gnostics library.