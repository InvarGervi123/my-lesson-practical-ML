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