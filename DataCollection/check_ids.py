import numpy as np
import requests
import os
import json
import time

UNFILERTED_ID_PATH = '../Steam Dataset Aggregation/data/userdata/'
PUBLIC_ID_PATH = '../Steam Dataset Aggregation/data/userdata/'
BATCH_INFO_PATH = '../Steam Dataset Aggregation/data/batch_schedules/batch_info.json'
ID_REQUEST_URL = 'http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/'
API_KEY = '4ACBCC16077837AE6E9FACE79A2B5683'
SAMPLE_ID = '76561198020265728'
PARAMS = {'key':API_KEY,
        'steamids': None}
#,
        # 'format':'json' 
BATCH_FMT = {'01' : {'batch_range' : [], 'batch_sizes' : [], 'current_idx' : None, 'batch_counter':None}}

# need to include the case for when the batch is finished running
# then need to put it on my windows machine so that it can make an attempt tomorrow 

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
    with open(BATCH_INFO_PATH, 'r') as f:
        batch_info = json.load(f)
    id_range = batch_info[vm_id]['batch_range']
    index = batch_info[vm_id]['current_idx']
    batch_sizes = batch_info[vm_id]['batch_sizes']
    batch_counter = batch_info[vm_id]['batch_counter']
    return id_range, index, batch_sizes, batch_counter

def subset_array(start, unchecked_ids):
    """
    Subsets the array and creates a comma separated list of ids based off of start and stop.
    
    Args:
        start (int) : The starting point to slice 100 values from.
        unchecked_ids (np.array) : The numpy array containing all possible Ids.
    Returns:
        ids (str) : A comma separated string of 100 ids.
    """
    subset = unchecked_ids[start:start+100]
    return ','.join(map(str, subset))

def send_ids_request(ids, counter, sleep_timer=60):
    """
    Send a get request for 100 Steam Ids to determine if they are public or not.

    Args:
        ids (str) : Ids to pass as part of the get request.

    Returns:
    public_ids (np.array) : Array containing public ids as int64s.
    """
    PARAMS['steamids'] = ids
    r = requests.get(ID_REQUEST_URL, params=PARAMS)
    try:
        return r.json()
    except json.decoder.JSONDecodeError:
        print("-------------")
        print(r.content)
        print("-------------")
        print(f"Request #{counter} caused rate limiting")
        if sleep_timer == 480:
            exit()
        time.sleep(sleep_timer)
        return send_ids_request(ids, counter, sleep_timer * 2)
        
# request # 4 caused rate limiting under current circumstance, meaning that around 24 seconds had passed before a rate limit got placed on me :<
    # request # 4 caused rate limiting again under a 10 second timer
        # under a 10.5 second timer, request #10 caused the rate issue
            # even after a sleep of 120 the rate limiter still said no
                # even after a sleep of 240 the rate limiter STILL said no, wowie zowie 

def fetch_mask(index, batch_size):
    """
    Initializes a mask array of booleans to determine visibility of Steam Ids.
    
    Args:
        index (int) : The starting index from the batch instructions to determine if a new mask array needs to be made.
        id_range (str) : The range of values being covered used to access a previously made mask file.
    
    Returns:
        mask_array (np.array) : The mask array for the id_range.
    """
    # if index != 0:
    #     mask_fname = f'{index}to{index+batch_size}' + 'mask.npy'
    #     rel_path = os.path.join(PUBLIC_ID_PATH, mask_fname)
    #     with open(rel_path, 'rb') as f:
    #         mask_array = np.load(f)
    # else:
    mask_array = np.empty(batch_size, dtype=bool)
    return mask_array

def fetch_unfiltered_ids(id_range):
    """
    Initializes a numpy array containing all userIds for a given id range.

    Args:
        id_range (str) : The range of values being covered in the given batch.

    Returns:
        unfilitered_id_array (np.array) : An array containing all unfiiltered Ids for a given range.
    """
    id_fname = id_range + 'ids.npy'
    rel_path = os.path.join(UNFILERTED_ID_PATH,id_fname)
    with open(rel_path, 'rb') as f:
        unfiltered_id_array = np.load(f, allow_pickle=False)
    return unfiltered_id_array

def parse_request(response, mask_array, start):
    """
    Parses the get request for Steam ids that are public, updating the mask array as necessary.

    Args:
        response (json) : The response from the get request.
        mask_array (np.array) : The mask array that gets updated.
        start (int) : The starting point in the unfiltered Ids that gets used to index the mask array.
    
    Return:
        mask_array (np.array) : The mask array after being updated with new positions.
    """
    if response:
        respo = response['response']
        players = respo['players']
        for number, player in enumerate(players):
            try:
                if player['communityvisibilitystate'] == 3:
                    mask_array[start + number] = True
            except:
                print(f"Deleted steam ID found at index {start + number}")
    return mask_array

def process_batch(id_range, index, batch_size):
    """
    Batch processes IDs for given set of batch instructions.

    Args:
        id_range (str) : The range of values being covered in the given batch.
        index (int) : The starting index in the array of IDs for this given batch.
        batch_size (int) : The batch size for this given batch.

    Returns:
        mask_array (np.array) : The numpy mask array updated with new information given the previous batch.
    """
    mask_array = fetch_mask(index, batch_size)
    unfiltered_ids = fetch_unfiltered_ids(id_range)
    counter = 1
    batch_index = 0
    for start in range(index, index+batch_size, 100):

        if counter == 1:
            begin = time.time()
        if counter % 8 == 0:
            threeMinGate = begin + 190
            now = time.time()
            wait_time = threeMinGate-now
            if wait_time > 0:
                time.sleep(wait_time)
            else:
                time.sleep(60)
            begin = time.time()
        
        ids = subset_array(start, unfiltered_ids)
        response = send_ids_request(ids, counter)

        mask_array = parse_request(response, mask_array, batch_index)
        batch_index += 100
        counter += 1
        time.sleep(11)
        
    return mask_array

def output_to_file(mask_array, index, batch_size):
    """
    Outputs a mask array to a file.

    Args: 
        mask_array (np.array) : The mask array containing publicity data.
        id_range (np.array) : The range of IDs being ocv
    """
    
    with open(PUBLIC_ID_PATH + f'{index}to{index+batch_size}mask.npy', 'wb') as f:
        np.save(f, mask_array)

# now we have to reconsider indexing for future batches, meaning we take our start index and thats the index for the 
    # main array etc, but I guess we also need to keep track of a mask index, which starts at 0 and then every iteration
        # we just add 100 and count off of that + number in enumerate

def read_mask(id_range):
    """
    Not yet implemented
    """
    with open(PUBLIC_ID_PATH + f'{id_range}mask.npy', 'wb') as f:
        mask_array = np.load(f)
    
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
    BATCH_FMT["01"].update({
        'batch_range': id_range,
        'batch_sizes': batch_sizes,
        'current_idx': index + batch_size,
        'batch_counter': batch_counter + 1
    })

    with open(BATCH_INFO_PATH, 'w') as f:
        json.dump(BATCH_FMT, f)

for _ in range(0, 15):
    start = time.time()
    id_range, index, batch_sizes, batch_counter = fetch_batch_info(vm_id="01")
    print(fetch_batch_info(vm_id="01"))
    batch_size = batch_sizes.pop(0)
    print(f"Starting batch for {id_range}, with batch_size of {batch_size}, this is batch # {batch_counter}")
    mask_array = process_batch(id_range, index, batch_size)
    update_batch_info(id_range, index, batch_sizes, batch_size, batch_counter)

    output_to_file(mask_array, index, batch_size)

# 4 seconds for 1000 user requests, or 4 seconds for 10 calls
# it takes around 8.3 hours of runtime to finish requesting all under a batch size of 10,000, the runtime for 10,000 userIDs is around 31 seconds, or 31 seconds for 100 calls
