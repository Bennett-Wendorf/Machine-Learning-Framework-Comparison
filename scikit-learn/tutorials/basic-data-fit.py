from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
model = RandomForestClassifier(random_state=0)

x = [[1, 2, 3], # 2 samples, 3 features
    [11, 12, 13]]

y = [0, 1]  # classes of each sample

# Train the model
model.fit(x, y)

print(model.predict(x))
print(model.predict([[4, 5, 6], [14, 15, 16]]))