# First Machine Learning Project – Apartment Price Classification

This project is my first clear and practical introduction to Machine Learning using Python and scikit-learn on EndeavourOS with Visual Studio Code.

The goal was to understand how AI actually works internally, not just use it.

This example uses apartment sizes and teaches the model to classify price categories.

---

# Example

Training data:

Size (m²) → Category

30 → very cheap  
45 → cheap  
50 → little cheap  
60 → little expensive  
75 → expensive  
90 → very expensive  

After training, the model can predict new inputs:

Size: 70 m²  
Prediction: expensive

---

# Code

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [30],
    [45],
    [50],
    [60],
    [75],
    [90]
])

y = np.array([
    "very cheap",
    "cheap",
    "little cheap",
    "little expensive",
    "expensive",
    "very expensive"
])

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

size = np.array([[70]])
prediction = model.predict(size)

print("Size:", size[0][0], "m²")
print("Category:", prediction[0])
```

---

# What I Understood

This example helped me understand the real connection between AI and mathematics.

Machine Learning works using vectors.

Each input is a vector.

Example:

x = [70]

The model learns relationships between input vectors and output categories.

It does not store answers directly.

Instead, it learns mathematical relationships.

When a new vector arrives, the model finds the closest relationship and predicts the most likely result.

This is directly connected to mathematics learned in school:

• vectors  
• functions  
• distances  
• mappings  

Machine Learning is essentially a function:

f(x) → y

---

# Connection to Large AI Models

This example uses only a few vectors.

Large AI models use billions of numerical parameters.

These parameters represent mathematical relationships.

This explains why large AI models require significant memory and computation.

The larger the number of parameters, the more relationships the model can represent.

---

# Why Model Sizes Differ

Models can have the same structure but different sizes because of numerical precision.

Examples:

float32 → larger size, higher precision  
float16 → smaller size  
int8 / int4 → compressed, much smaller  

Lower precision reduces memory and compute requirements.

---

# Key Realization

AI is not magic.

It is mathematics applied to vectors at scale.

This simple apartment example made the concept clear and practical.

---

# Environment

OS: EndeavourOS  
CPU: Intel Core i7  
GPU: Intel Iris Xe  
Editor: Visual Studio Code  

Libraries:

numpy  
scikit-learn

---

# Purpose

This project represents my first real understanding of how Machine Learning works internally.

