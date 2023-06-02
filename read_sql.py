import sqlite3
import numpy as np
import torch
from copy import deepcopy

def reward_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("C:/500/reward_list_500.db")
    
    # Prepare the SQL statement
    stmt = "SELECT * FROM reward_list"
    
    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()
    
    # Close the connection
    conn.close()
    
    # Convert the data to a NumPy array
    flattened_data = np.array(data).flatten()

    array = np.reshape(flattened_data, (size, 1))

    return array

def initial_states_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("C:/500/initial_states_list_500.db")
    
    # Prepare the SQL statement
    stmt = "SELECT * FROM initial_states_list"
    
    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()
    
    # Close the connection
    conn.close()
    
    # Convert the data to a NumPy array
    flattened_data = np.array(data).flatten()

    array = np.reshape(flattened_data, (size, 7))

    return array

def weights_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("C:/500/weights_list_360.db")
    
    # Prepare the SQL statement
    stmt = "SELECT * FROM weights_list"
    
    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()
    
    # Close the connection
    conn.close()
    
    # Convert the data to a NumPy array
    flattened_data = np.array(data).flatten()

    array = np.reshape(flattened_data, (size, 3))

    return array

def costmap_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("C:/500/costmap_list_500.db")    

    # Prepare the SQL statement
    stmt = "SELECT * FROM costmap_list"

    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()

    # Close the connection
    conn.close()

    # Extract the data as lists of bytes
    byte_data = [list(row[0]) for row in data]

    # Convert the data to a NumPy array
    # flattened_data = np.array(byte_data, dtype=np.uint8).flatten()
    flattened_data = np.array(byte_data).flatten()

    # Reshape the flattened data to the desired shape
    array = np.reshape(flattened_data, (size, 40, 40))

    return array


def delete_data(rewards,costweights,states,costmaps):
    a = np.where(rewards==0)
    b = np.where(rewards==100)
    diff = len(a[0])-len(b[0])
    c = range(diff)

    rewards_ = np.delete(rewards,a[0][0:diff])
    costweights_ = np.delete(costweights,a[0][0:diff],0)
    states_ = np.delete(states,a[0][0:diff],0)
    costmaps_ = np.delete(costmaps,a[0][0:diff],0)

    return costmaps_, states_, costweights_, rewards_


def inputs(costmaps,states,costweights):
    new_costmaps = costmaps
    new_states = states
    new_costweights = costweights
    for i in range(len(costweights)-1):
        new_costmaps = np.append(new_costmaps,costmaps,axis=0)
        new_states = np.append(new_states,states,axis=0)

    for z in range(len(states)-1):
        new_costweights = np.append(new_costweights,costweights,axis=0)

    return new_costmaps, new_states, new_costweights

def normalize_data(states):

    new_states = deepcopy(states)
    high_x = 19
    high_y = 9
    high_goal_x = 20
    high_goal_y = 10
    high_theta = 2 * np.pi
    low_theta = -2 * np.pi
    min_v = 0
    max_v = 0.1
    min_w = -np.pi/4
    max_w = np.pi/4

    for i in range(len(states)):
        new_states[i][0] = states[i][0] / high_x
        new_states[i][1] = states[i][1] / high_y
        new_states[i][2] = (states[i][2] - low_theta) / (high_theta - low_theta)
        new_states[i][3] = states[i][3] / high_goal_x
        new_states[i][4] = states[i][4] / high_goal_y
        new_states[i][5] = (states[i][5] - min_v) / (max_v - min_v)
        new_states[i][6] = (states[i][6] - min_w) / (max_w - min_w)

    return new_states

# Usage example
# size = 10000
# result = reward_sqlite(size)
# print(result.shape)
# result = initial_states_sqlite(size)
# print(result.shape)
# result = weights_sqlite(size)
# print(result.shape)
# result = costmap_sqlite(size)
# print(result.shape)
# print(result)
