import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(4*4*32 + 7 + 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, costmaps, states, costweights):
        x = self.relu(self.conv1(costmaps))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 4*4*32)
        x = torch.cat([x, states,costweights], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

def train_net(train_data, batch_size=32, num_epochs=10, lr=0.01):

    # Create network and optimizer
    net = MyNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    x_train = train_data[0]
    y_train = train_data[1]

    # Train network
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in tqdm(range(0, len(x_train), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Get batch of inputs and targets
            inputs = x_train[i:i+batch_size].to(device)
            targets = y_train[i:i+batch_size].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs[:, :1], inputs[:, 1:8], inputs[:, 8:])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % (batch_size * 50) == 0:
                print('[Epoch %d, Batch %d/%d] Loss: %.3f' %
                      (epoch + 1, (i + 1) / batch_size, len(x_train) / batch_size, running_loss / (batch_size * 50)))
                running_loss = 0.0
    
    return net

def test_net(net, test_data, batch_size=32):
    net.eval() # switch to evaluation mode
    correct = 0
    total = 0

    input_batch = test_data[0]
    target_batch = test_data[1]

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            # Get batch of inputs and targets
            input_batch = test_data[i:i+batch_size]
            target_batch = test_data[i:i+batch_size]

            # Convert input data to PyTorch tensors
            x = torch.from_numpy(np.array(input_batch[:, :1])).float()
            y = torch.from_numpy(np.array(input_batch[:, 1:8])).float()
            c = torch.from_numpy(np.array(input_batch[:, 8:])).float()

            # Move tensors to GPU if available
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                c = c.cuda()
                target_batch = target_batch.cuda()

            # Forward pass
            output = net(x, y, c)
            _, predicted = torch.max(output.data, 1)

            # Compute accuracy
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum().item()

    accuracy = 100.0 * correct / total
    print('Test Accuracy: %d %%' % (accuracy))
    return accuracy

def utility(costmaps, states, costweights, rewards):

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(costmaps, states, costweights, rewards, test_size=0.2, random_state=42)

    # Convert input data to PyTorch tensors
    x_train = torch.from_numpy(np.array(x_train)).float()
    x_test = torch.from_numpy(np.array(x_test)).float()
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_test = torch.from_numpy(np.array(y_test)).long()

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return train_data, test_data


# costmaps = 
# states = 
# costweights = 
# rewards =  

def main(costmaps, states, costweights, rewards):
    
    # Preprocess data and split into train/test sets
    train_data, test_data = utility(costmaps, states, costweights, rewards)

    # Train network
    net = train_net(train_data)

    # Test network
    test_accuracy = test_net(net, test_data)

    # Save trained network
    torch.save(net.state_dict(), 'trained_network.pth')  




# In order to load the saved network use the code below !!

# Create an instance of MyNet
# net = MyNet()

# Load the saved state dictionary
# net.load_state_dict(torch.load('trained_network.pth'))