import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from read_sql import reward_sqlite, costmap_sqlite, weights_sqlite, initial_states_sqlite, delete_data, inputs,normalize_data
from sklearn.metrics import f1_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(4*4*32 + 7 + 3, 128) #128 di denemek için arttırdım ve azalttım
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 64) #denemek için ekledim
        self.fc4 = nn.Linear(64, 3) #for softmax

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # self.costmaps_normalizer = nn.BatchNorm2d(1)  # Normalizes costmaps
        # self.states_normalizer = nn.BatchNorm1d(7)  # Normalizes states
        # self.costweights_normalizer = nn.BatchNorm1d(3)  # Normalizes costweights

    def forward(self, costmaps, states, costweights):
        output = torch.tensor([])
        # output = []
        # costmaps = self.costmaps_normalizer(costmaps)
        # states = self.states_normalizer(states)
        # costweights = self.costweights_normalizer(costweights)
        for i in range(len(states)):
            x = self.relu(self.conv1(costmaps[i]))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = x.view(-1, 4*4*32)

            states_i = states[i].unsqueeze(0)

            for costweight in costweights:

                costweight_tensor = torch.tensor(costweight)
                costweight_tensor = costweight_tensor.unsqueeze(0)

                # Concatenate the inputs along the channel dimension
                x_combined = torch.cat([x, states_i,costweight_tensor], dim=1)
                x_combined = x_combined.float()
                
                # x_combined = x_combined.view(-1, 4*4*32 + 7 + 3)

                x_combined = self.relu(self.fc1(x_combined))
                x_combined = self.relu(self.fc2(x_combined))
                x_combined = self.relu(self.fc3(x_combined))
                x_combined = self.fc4(x_combined)
                output = torch.cat([output, x_combined],dim=0)
                # output.append(x_combined)

        # output = torch.stack(output, dim=0)
        return output

    # def forward(self, costmaps, states, costweights):

    #     x = self.relu(self.conv1(costmaps))
    #     x = self.relu(self.conv2(x))
    #     x = self.relu(self.conv3(x))
    #     x = self.relu(self.conv4(x))
    #     x = x.view(-1, 4*4*32)



    #     # Concatenate the inputs along the channel dimension
    #     x_combined = torch.cat([x, states,costweights], dim=1)
    #     # x_combined = x_combined.float()


    #     x_combined = self.relu(self.fc1(x_combined))
    #     x_combined = self.relu(self.fc2(x_combined))
    #     x_combined = self.relu(self.fc3(x_combined))
    #     x_combined = self.fc4(x_combined)


    #     return x_combined


def init_weights(m):
    if isinstance(m,nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight,
                               a=0, mode="fan_in",
                               nonlinearity="relu")
        # torch.nn.init.ones_(m.weight)


def train_net(train_data, val_data, costweights, batch_size, num_epochs=100, lr=0.001):  #lr=0.001

    # Create network and optimizer
    net = MyNet()
    net.load_state_dict(torch.load('trained_network_1.pth')) #train edilmiş parametreler ile initialize ettim
    # net.apply(init_weights) 

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=25, verbose=True)
    # class_weights = torch.tensor([1.0, 82/9411, 82/507]) #for 10000
    # class_weights = torch.tensor([1.0, 752/91109, 752/8139]) #for 100000
    # class_weights = torch.tensor([1.0, 4588/452681, 4588/42731]) #for 500000
    # class_weights = torch.tensor([13649/70133, 13649/96218, 1]) #for 180000
    # class_weights = torch.tensor([29219/117837, 29219/212944, 1]) #for 360000
    # class_weights = torch.tensor([18/714, 1, 18/6468]) #for 7200
    class_weights = torch.tensor([25/714, 1, 25/6468]) #for 7200

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    x_train, s_train, y_train = train_data
    x_val, s_val, y_val = val_data

    x_train = x_train.to(device)
    s_train = s_train.to(device)
    # w_train = w_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    s_val = s_val.to(device)
    # w_val = w_val.to(device)
    y_val = y_val.to(device)

    best_f1 = 0.0
    epochs_without_improvement = 0
    max_epochs_without_improvement = 150

    # Train network
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training phase
        net.train()
        for i in tqdm(range(0, len(x_train), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):

            if i + batch_size <= len(x_train):
            # Get batch of inputs and targets
                costmaps = x_train[i:i+batch_size].unsqueeze(1)
                states = s_train[i:i+batch_size]
                # weights = w_train[i:i+batch_size]
                target_batch_size = batch_size * len(costweights)
                targets = y_train[i:i+target_batch_size]
                targets = targets.squeeze()

            else:
                # Handle the remaining samples
                remaining_samples = len(x_train) - i
                costmaps = x_train[i:].unsqueeze(1)
                states = s_train[i:]
                targets = y_train[i:i+remaining_samples*len(costweights)]
                targets = targets.squeeze()


            costmaps = costmaps.to(device)
            states = states.to(device)
            # costweights = costweights.to(device)
            targets = targets.to(device)
            class_labels = torch.zeros_like(targets)
            class_labels[targets==-30] = 0
            class_labels[targets==0] = 1
            class_labels[targets==100] = 2

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(costmaps, states, costweights)
            loss = criterion(outputs.view(-1, 3), class_labels.view(-1))
            # loss = criterion(outputs.squeeze(), class_labels)

            # loss_function = torch.nn.NLLLoss()
            # loss_2 = loss_function(outputs.squeeze(), class_labels)
            # print("Calculated Loss:", loss_2.item())


            # class_labels = class_labels.float()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * costmaps.size(0)

        # Print statistics
        epoch_loss = running_loss / len(x_train)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
    
        #Validation phase
        net.eval()
        running_val_loss = 0.0
        predicted_labels = np.array([])
        true_labels = np.array([])

        with torch.no_grad():
            for i in range(0, len(x_val), batch_size):

                if i + batch_size <= len(x_val):
                    costmaps = x_val[i:i+batch_size].unsqueeze(1)
                    states = s_val[i:i+batch_size]
                    # weights = w_val[i:i+batch_size]
                    target_batch_size = batch_size * len(costweights)
                    targets = y_val[i:i+target_batch_size]
                    targets = targets.squeeze()

                else:
                    # Handle the remaining samples
                    remaining_samples = len(x_val) - i
                    costmaps = x_val[i:].unsqueeze(1)
                    states = s_val[i:]
                    targets = y_val[i:i+remaining_samples*len(costweights)]
                    targets = targets.squeeze()

                class_labels = torch.zeros_like(targets)
                class_labels[targets==-30] = 0
                class_labels[targets==0] = 1
                class_labels[targets==100] = 2

                outputs = net(costmaps, states, costweights)
                val_loss = criterion(outputs,class_labels)
                predicted = torch.argmax(outputs, dim=1)

                predicted_labels = np.append(predicted_labels, predicted.numpy())
                true_labels = np.append(true_labels, class_labels.numpy())

                # Accumulate validation loss
                running_val_loss += val_loss.item() * costmaps.size(0)

        # Calculate average validation loss
        avg_val_loss = running_val_loss / len(x_val)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Print learning rate
        print(f"Epoch {epoch+1} Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Print or store the average validation loss
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        f1_scores = []
        for label in range(3):
            mask = true_labels == label
            f1 = f1_score(true_labels[mask], predicted_labels[mask], average='weighted')
            f1_scores.append(f1)
            print(f"Epoch {epoch+1} F1 Score (Label {label}): {f1:.4f}")

        avg_f1 = np.mean(f1_scores)
        print(f"Epoch {epoch+1} Average F1 Score: {avg_f1:.4f}")

        # Early stopping based on F1 score improvement
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            epochs_without_improvement = 0
            # torch.save(net.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == max_epochs_without_improvement:
                print("Early stopping triggered!")
                break

    return net

def test_net(net, test_data, costweights, batch_size):
    net.eval() # switch to evaluation mode
    correct = 0
    total = 0
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    class_weights = torch.tensor([13359/54843, 13359/111807, 1])

    x_test, s_test, y_test = test_data

    predicted_labels = np.array([])
    true_labels = np.array([])

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            # Get batch of inputs and targets
            if i + batch_size <= len(x_test):
                costmaps = x_test[i:i+batch_size].unsqueeze(1)
                states = s_test[i:i+batch_size]
                target_batch_size = batch_size * len(costweights)
                # weights = w_test[i:i+batch_size]
                targets = y_test[i:i+target_batch_size]
            else:
                # Handle the remaining samples
                remaining_samples = len(x_test) - i
                costmaps = x_test[i:].unsqueeze(1)
                states = s_test[i:]
                targets = y_test[i:i+remaining_samples*len(costweights)]

            costmaps = costmaps.to(device)
            states = states.to(device)
            # costweights = costweights.to(device)
            targets = targets.to(device)
            class_labels = torch.zeros_like(targets)
            class_labels[targets==-30] = 0
            class_labels[targets==0] = 1
            class_labels[targets==100] = 2

            # Convert input data to PyTorch tensors and move to GPU if available
            costmaps = torch.from_numpy(np.array(costmaps)).float()
            states = torch.from_numpy(np.array(states)).float()
            # costweights = torch.from_numpy(np.array(costweights)).float()
            # targets = torch.from_numpy(np.array(targets)).long()
            targets = torch.from_numpy(np.array(targets))

            # Forward pass
            outputs = net(costmaps, states, costweights)
            predicted = torch.argmax(outputs, dim=1)
            # _, predicted = torch.max(outputs, dim=2)
            class_labels = class_labels.squeeze()

            # Compute accuracy
            total += class_labels.size(0)
            correct += (predicted.squeeze() == class_labels).sum().item()

            # Collect predicted and true labels for F1 score calculation
            predicted_labels = np.append(predicted_labels, predicted.numpy())
            true_labels = np.append(true_labels, class_labels.numpy())

    accuracy = 100.0 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))

    # Calculate F1 score
    f1_scores = []
    for label in range(3):
        mask = true_labels == label
        f1 = f1_score(true_labels[mask], predicted_labels[mask], average='weighted')
        f1_scores.append(f1)
        print(f"F1 Score (Label {label}): {f1:.4f}")

    # target_names = ['obstacle', 'terminated', 'goal']
    # print(classification_report(true_labels,predicted_labels,sample_weight=class_weights))

    return accuracy


def utility(costmaps, states, rewards):
    # Split data into training and test sets
    x_train, x_test = train_test_split(costmaps, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(rewards, test_size=0.2, random_state=42)
    s_train, s_test = train_test_split(states, test_size=0.2, random_state=42)
    # w_train, w_test = train_test_split(costweights, test_size=0.2, random_state=42)

    # x_train, x_val, s_train, s_val, w_train, w_val, y_train, y_val = train_test_split(x_train, s_train, w_train, y_train, test_size=0.2, random_state=42)

    x_train, x_val, s_train, s_val = train_test_split(x_train, s_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)

    # Convert input data to PyTorch tensors
    x_train = torch.from_numpy(np.array(x_train)).float()
    x_test = torch.from_numpy(np.array(x_test)).float()
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_test = torch.from_numpy(np.array(y_test)).long()
    y_train = torch.from_numpy(np.array(y_train))
    y_test = torch.from_numpy(np.array(y_test))
    s_train = torch.from_numpy(np.array(s_train)).float()
    s_test = torch.from_numpy(np.array(s_test)).float()
    # w_train = torch.from_numpy(np.array(w_train)).float()
    # w_test = torch.from_numpy(np.array(w_test)).float()
    x_val = torch.from_numpy(np.array(x_val)).float()
    s_val = torch.from_numpy(np.array(s_val)).float()
    # w_val = torch.from_numpy(np.array(w_val)).float()
    y_val = torch.from_numpy(np.array(y_val)).long()

    train_data = (x_train, s_train, y_train)
    test_data = (x_test, s_test, y_test)
    val_data = (x_val, s_val, y_val)

    return train_data, test_data, val_data



# size = 10000
# size = 100000
# size = 500000
size = 20
# size = 1000
# batch_size = 128
batch_size = 2 #4 tü
# batch_size = 6
# batch_size = 50
weight_size = 360
reward_size = size * weight_size
# batch_size = 512
# batch_size = 1024
# batch_size = 2048
costmaps = costmap_sqlite(size)
states = initial_states_sqlite(size)
costweights = weights_sqlite(weight_size)
rewards =  reward_sqlite(reward_size)

states = normalize_data(states)

# costmaps, states, costweights = inputs(costmaps, states, costweights)
# costmaps, states, costweights, rewards = delete_data(rewards, costweights, states, costmaps)

def main(costmaps, states, costweights, rewards):
    
    # Preprocess data and split into train/test/validation sets
    train_data, test_data, val_data = utility(costmaps, states, rewards)

    # Train network
    net = train_net(train_data, val_data, costweights, batch_size)

    # Test network
    test_accuracy = test_net(net, test_data, costweights, batch_size)

    # Save trained network
    torch.save(net.state_dict(), 'trained_network_2.pth')  

main(costmaps, states, costweights, rewards)



# In order to load the saved network use the code below !!

# Create an instance of MyNet
# net = MyNet()

# Load the saved state dictionary
# net.load_state_dict(torch.load('trained_network.pth'))