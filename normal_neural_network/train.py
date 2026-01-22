from system_nn import SimpleNN

# XOR dataset
dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

model = SimpleNN(input_dim=2, hidden_dim=8, lr=0.1)

for epoch in range(2000):
    loss = 0
    for x, y in dataset:
        loss += model.train_step(x, y)
    if epoch % 200 == 0:
        print(epoch, loss)

# test
for x, y in dataset:
    print(x, model.forward(x))