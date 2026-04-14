import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('student_data.csv')

# Display dataset
print("Dataset:\n", data.head())

# Extract variables
X = data['hours'].values
Y = data['marks'].values

# Calculate mean
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Calculate slope (m) and intercept (b)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x) ** 2)

m = numerator / denominator
b = mean_y - (m * mean_x)

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Prediction function
def predict(x):
    return m * x + b

# Predict marks for 7.5 hours
hours = 7
predicted_marks = predict(hours)
print(f"Predicted Marks for {hours} hours: {predicted_marks}")

# Plotting
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, m*X + b, color='red', label='Regression Line')

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")

plt.legend()
plt.grid()
plt.savefig("graph.png")
plt.show()