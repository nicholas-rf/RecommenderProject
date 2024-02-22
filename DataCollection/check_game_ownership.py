import requests
import numpy as np
import json
import pandas as pd
import os
import time
UNFILERTED_ID_PATH = '../Recommender/data/userdata/'
CSV_PATH = '../Recommender/data/'
BATCH_INFO_PATH = '../Recommender/data/batch_schedules/batch_info.txt'
BATCH_FMT = {'01' : {'batch_range' : [], 'batch_sizes' : [], 'current_idx' : None, 'batch_counter':None}}
API_KEY = '4ACBCC16077837AE6E9FACE79A2B5683'
GET_GAMES_CALL = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
SAMPLE_ID = '76561198020265728'
PARAMS = {'key':API_KEY,
        'steamid': None,
        'format':'json'}


def update_batch_info(id_range, index, batch_sizes, batch_size, batch_counter):
    """
    Updates the batch information file for the specific instance with new instructions.

    Args:
        id_range (str) : The range of Ids being covered.
        index (int) : The current instructions starting index in the array of Ids.
        batch_sizes (int) : The batch sizes for runtime.
        batch_size (int) : The batch size that was just ran.
        batch_counter (int) : The counter signifying which batch is being ran.
    """
    # Use the standard batch format and .update to update the batch file with new information
    BATCH_FMT["01"].update({
        'batch_range': id_range,
        'batch_sizes': batch_sizes,
        'current_idx': index + batch_size,
        'batch_counter': batch_counter + 1
    })

    batch_as_string = str(BATCH_FMT)
    with open(BATCH_INFO_PATH, 'w') as f:
        f.write(batch_as_string)

def fetch_batch_info(vm_id):
    """
    Gets batch instructions and returns them for the specified virtual machine.

    Args:
        vm_id (str) : The id of the virtual machine.
    Returns:
        id_range (str) : The range of Ids being covered.
        index (int) : The starting index in the array of Ids.
        batch_sizes (int) : The batch sizes for runtime.
        batch_counter (int) : The counter signifying which batch is being ran.
    """
    # Open up the batch and read in all the necessary information
    with open(BATCH_INFO_PATH, 'r') as f:
        batch_info = eval(f.readline())
    id_range = batch_info[vm_id]['batch_range']
    index = batch_info[vm_id]['current_idx']
    batch_sizes = batch_info[vm_id]['batch_sizes']
    batch_counter = batch_info[vm_id]['batch_counter']
    return id_range, index, batch_sizes, batch_counter

def get_request(id, wait_time=180):
    """
    Sends a request to the steam API for a users owned games
    """
    # Set the parameter dictionary for the get request key for the ID that we will be checking
    PARAMS['steamid'] = id

    # Send the get request
    r = requests.get(GET_GAMES_CALL, params=PARAMS)
    try:
        # Attempt to return the json format of the response
        return r.json()
    except json.decoder.JSONDecodeError:
        # If there is an error, print it and recursively call the function with a sleep time to avoid rate-limiting
        print("-------------")
        print(r.content)
        print(f"Sleeping for {wait_time/60} minutes then trying again")
        print("-------------")
        return get_request(id, wait_time*2)
    
def fetch_ids(id_range):
    """
    Fetches a list of IDs to load and parse through.
    
    Args:
        id_list (string) : The list from which IDs are being gathered from.
    Returns:
        public_game_ids (np.array) : The np array of 100,000 public IDs to be checked for games ownership.
    """
    # Fetches the list of IDs for the given id_range
    rel_path = os.path.join(UNFILERTED_ID_PATH, id_range+"ids.npy") 
    with open(rel_path, 'rb') as f:
        return np.load(f)


def parse_request(response,id):
    """
    Parses the response to the request for games ownership creating a dataframe row containing all owned games, playtime and last playtime.

    Args:
        response (r.json()) : The response from the get request containing owned games info.
        id (str) : The players id associated with the get request.
    Returns:
        A dataframe (pd.DataFrame) : A dataframe containing the dictionary of data created for the player, given that it was available. 
    """
    try:
        # Attempt to parse through the data returned by the get request, getting the users id, total owned games, appids, playtime and last played games
        full_response = response['response']
        total_owned_games = full_response['game_count']
        owned_apps = full_response['games']
        data_dict = {"player_id":id, "total_owned": total_owned_games, "appids":[[]], "playtime":[[]], "last_played":[[]]}
        for app in owned_apps:
            data_dict["appids"][0].append(app['appid'])
            data_dict["playtime"][0].append(app['playtime_forever'])
            data_dict["last_played"][0].append(app['rtime_last_played'])

        # With the dictionary filled with the players data, output it as a pandas dataframe
        return pd.DataFrame(data_dict)
    except:
        return pd.DataFrame()
        

def batch_process(ids, index, batch_size):
    """
    Processes a batch given its instructions.

    Args:
        id_range (str) : The range of Ids being covered.
        index (int) : The current instructions starting index in the array of Ids.
        batch_size (int) : The batch size that is being ran.
        batch_counter (int) : The counter signifying which batch is being ran.

    Returns:
        complete_frame (pd.DataFrame) : A dataframe containing rows with users and their owned games.
    """

    # Initialize some variables for later use
    complete_frame = pd.DataFrame()
    counter = 0
    timer_start = time.time()
    print(f"Requests starting")

    # Iterate over all players in the specified batch_size
    for i in range(index, index+batch_size):

        # If we reach a batch number that is a multiple of 200, check to see if it has been 5 minutes since sending the 1st of 200 requests to see if waiting to avoid rate-limiting needs to occur
        if counter % 200 == 0 or counter == 0:
            difference = time.time() - timer_start
            if difference < 300 and counter != 0:
                time.sleep(difference)
            timer_start = time.time()
        
        # Send the get request for the ID and create a dataframe from the response
        response = get_request(ids[i])
        df = parse_request(response, ids[i])

        # If our dataframe is empty sleep for 1.6 seconds
        if df.empty:
            time.sleep(1.6)  
        
        # Otherwise concatenate the new dataframe onto the complete dataframe, increment the batch_counter and sleep for 1.6 seconds
        else:
            complete_frame = pd.concat([complete_frame, df], ignore_index=True, axis=0)
            time.sleep(1.6)
        counter += 1
    
    # Print out the memory usage of the dataframe for testing purposes
    memory_usage = complete_frame.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage of DataFrame: {memory_usage:.2f} MB")
    return complete_frame

def main():
    for batch in range(10):
        # Get the batch information
        id_range, index, batch_sizes, batch_counter = fetch_batch_info("01")
        
        # Get the size of the batch to be checked
        batch_size = batch_sizes.pop(0)
        
        # Load in the ID array specified for the batch
        ids=fetch_ids(id_range)
        
        # Process the batch
        df = batch_process(ids, index, batch_size)

        # Create a relative path for the csv and export it if the dataframe has rows
        if not df.empty:
            rel_path = f"{index}to{index+batch_size}.csv"
            df.to_csv(os.path.join(CSV_PATH + rel_path))

        # Update the batch info
        update_batch_info(id_range, index, batch_sizes, batch_counter)
        
if __name__ == "__main__":
    main()
