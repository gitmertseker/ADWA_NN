import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from read_sql import reward_sqlite, costmap_sqlite, weights_sqlite, initial_states_sqlite

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
        self.fc4 = nn.Linear(64, 3) #for softmax

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.costmaps_normalizer = nn.BatchNorm2d(1)  # Normalizes costmaps
        self.states_normalizer = nn.BatchNorm1d(7)  # Normalizes states
        self.costweights_normalizer = nn.BatchNorm1d(3)  # Normalizes costweights

    def forward(self, costmaps, states, costweights):

        costmaps = self.costmaps_normalizer(costmaps)
        states = self.states_normalizer(states)
        costweights = self.costweights_normalizer(costweights)

        x = self.relu(self.conv1(costmaps))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 4*4*32)

        # Concatenate the inputs along the channel dimension

        x = torch.cat([x, states,costweights], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x


def train_net(train_data, batch_size, num_epochs=200, lr=0.0005):

    # Create network and optimizer
    net = MyNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    class_weights = torch.tensor([1.0, 82/9411, 82/507])
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    x_train, s_train, w_train, y_train = train_data

    x_train = x_train.to(device)
    s_train = s_train.to(device)
    w_train = w_train.to(device)
    y_train = y_train.to(device)


    # Train network
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in tqdm(range(0, len(x_train), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):

            # Get batch of inputs and targets
            costmaps = x_train[i:i+batch_size].unsqueeze(1)
            states = s_train[i:i+batch_size]
            weights = w_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]
            targets = targets.squeeze()


            costmaps = costmaps.to(device)
            states = states.to(device)
            weights = weights.to(device)
            targets = targets.to(device)
            class_labels = torch.zeros_like(targets)
            class_labels[targets==-30] = 0
            class_labels[targets==0] = 1
            class_labels[targets==100] = 2

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(costmaps, states, weights)
            loss = criterion(outputs, class_labels)

            # class_labels = class_labels.float()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * costmaps.size(0)

        # Print statistics
        epoch_loss = running_loss / len(x_train)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
    
    return net

def test_net(net, test_data, batch_size):
    net.eval() # switch to evaluation mode
    correct = 0
    total = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    x_test, s_test, w_test, y_test = test_data

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            # Get batch of inputs and targets
            costmaps = x_test[i:i+batch_size].unsqueeze(1)
            states = s_test[i:i+batch_size]
            weights = w_test[i:i+batch_size]
            targets = y_test[i:i+batch_size]

            costmaps = costmaps.to(device)
            states = states.to(device)
            weights = weights.to(device)
            targets = targets.to(device)
            class_labels = torch.zeros_like(targets)
            class_labels[targets==-30] = 0
            class_labels[targets==0] = 1
            class_labels[targets==100] = 2

            # Convert input data to PyTorch tensors and move to GPU if available
            costmaps = torch.from_numpy(np.array(costmaps)).float()
            states = torch.from_numpy(np.array(states)).float()
            weights = torch.from_numpy(np.array(weights)).float()
            # targets = torch.from_numpy(np.array(targets)).long()
            targets = torch.from_numpy(np.array(targets))

            # Forward pass
            outputs = net(costmaps, states, weights)
            _, predicted = torch.max(outputs.data, 1)
            class_labels = class_labels.squeeze()
            # Compute accuracy
            # total += targets.size(0)
            total += class_labels.size(0)
            # correct += (predicted == targets).sum().item()
            correct += (predicted == class_labels).sum().item()

    accuracy = 100.0 * correct / total
    # print('Test Accuracy: %d %%' % (accuracy))
    print('Test Accuracy: {:.2f}%'.format(accuracy))
    return accuracy


def utility(costmaps, states, costweights, rewards):
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(costmaps, rewards, test_size=0.2, random_state=42)
    s_train, s_test = train_test_split(states, test_size=0.2, random_state=42)
    w_train, w_test = train_test_split(costweights, test_size=0.2, random_state=42)

    # Convert input data to PyTorch tensors
    x_train = torch.from_numpy(np.array(x_train)).float()
    x_test = torch.from_numpy(np.array(x_test)).float()
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_test = torch.from_numpy(np.array(y_test)).long()
    y_train = torch.from_numpy(np.array(y_train))
    y_test = torch.from_numpy(np.array(y_test))
    s_train = torch.from_numpy(np.array(s_train)).float()
    s_test = torch.from_numpy(np.array(s_test)).float()
    w_train = torch.from_numpy(np.array(w_train)).float()
    w_test = torch.from_numpy(np.array(w_test)).float()

    train_data = (x_train, s_train, w_train, y_train)
    test_data = (x_test, s_test, w_test, y_test)

    return train_data, test_data



size = 10000
# batch_size = 256
batch_size = 1024
costmaps = costmap_sqlite(size)
states = initial_states_sqlite(size)
costweights = weights_sqlite(size)
rewards =  reward_sqlite(size)

def main(costmaps, states, costweights, rewards):
    
    # Preprocess data and split into train/test sets
    train_data, test_data = utility(costmaps, states, costweights, rewards)

    # Train network
    net = train_net(train_data, batch_size)

    # Test network
    test_accuracy = test_net(net, test_data, batch_size)

    # Save trained network
    torch.save(net.state_dict(), 'trained_network.pth')  

main(costmaps, states, costweights, rewards)



# In order to load the saved network use the code below !!

# Create an instance of MyNet
# net = MyNet()

# Load the saved state dictionary
# net.load_state_dict(torch.load('trained_network.pth'))