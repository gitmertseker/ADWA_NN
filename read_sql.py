import sqlite3
import numpy as np
import torch

def reward_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("C:/100000/reward_list_100000.db")
    
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
    conn = sqlite3.connect("C:/100000/initial_states_list_100000.db")
    
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
    conn = sqlite3.connect("C:/100000/weights_list_100000.db")
    
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
    conn = sqlite3.connect("C:/100000/costmap_list_100000.db")    

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
