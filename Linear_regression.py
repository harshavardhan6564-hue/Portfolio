
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0],
                  [5.0]])

Y = torch.tensor([[2.0],
                  [4.0],
                  [6.0],
                  [8.0],
                  [10.0]])

model = nn.Linear(1, 1)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.02)

epochs = 1000

print("Training Started")

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.6f}")

print("\nTraining Completed")

weight = model.weight.item()
bias = model.bias.item()

print("Learned Weight:", weight)
print("Learned Bias:", bias)

test_data = torch.tensor([[6.0],
                          [7.0],
                          [8.0],
                          [10.0]])

print("\nTest Predictions")

with torch.no_grad():

    test_predictions = model(test_data)

    for i in range(len(test_data)):

        input_value = test_data[i].item()
        predicted_value = test_predictions[i].item()
        actual_value = 2 * input_value

        print("Input:", input_value)
        print("Predicted:", predicted_value)
        print("Actual:", actual_value)
        print("-------------------")

print("Program Completed")
