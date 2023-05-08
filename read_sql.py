import sqlite3
import numpy as np

def reward_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("D:/python/adwa_nn/10000_with_reward/10000_with_reward/reward_list_10000.db")
    
    # Prepare the SQL statement
    stmt = "SELECT * FROM reward_list"
    
    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()
    
    # Close the connection
    conn.close()
    
    # Convert the data to a NumPy array
    flattened_data = np.array(data).flatten()

    array = np.reshape(flattened_data, (size, 1))

    return array.tolist()

def initial_states_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("D:/python/adwa_nn/10000_with_reward/10000_with_reward/initial_states_list_10000.db")
    
    # Prepare the SQL statement
    stmt = "SELECT * FROM initial_states_list"
    
    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()
    
    # Close the connection
    conn.close()
    
    # Convert the data to a NumPy array
    flattened_data = np.array(data).flatten()

    array = np.reshape(flattened_data, (size, 7))

    return array.tolist()   

def weights_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("D:/python/adwa_nn/10000_with_reward/10000_with_reward/weights_list_10000.db")
    
    # Prepare the SQL statement
    stmt = "SELECT * FROM weights_list"
    
    # Execute the SQL statement and fetch the data
    data = conn.execute(stmt).fetchall()
    
    # Close the connection
    conn.close()
    
    # Convert the data to a NumPy array
    flattened_data = np.array(data).flatten()

    array = np.reshape(flattened_data, (size, 3))

    return array.tolist()

def costmap_sqlite(size):
    # Connect to the database
    conn = sqlite3.connect("D:/python/adwa_nn/10000_with_reward/10000_with_reward/costmap_list_10000.db")    

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

    return array.tolist()



# Usage example
# size = 10000
# result = reward_sqlite(size)
# result = initial_states_sqlite(size)
# result = weights_sqlite(size)
# result = costmap_sqlite(size)
# print(result)
