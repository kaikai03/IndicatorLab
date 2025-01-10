import subprocess
from pymongo import MongoClient
from datetime import datetime
import sys

port = 23333
dbpath = 'mongodb://localhost'+ ':' + str(port) +'/'
out_path = 'G:\\dbbk'



def full_backup():
    print('start full_backup:')
    now_time = datetime.now()
    folder_name = "full" + now_time.strftime('%Y%m%d%H%M%S')
    
    cmd = "mongodump --port=" \
    + str(port) \
    + " --gzip " \
    + " --out " \
    + out_path + "\\"+ folder_name +"\\"


    res = subprocess.call(cmd, shell=True)

    if res==0:
        client = MongoClient(dbpath)
        quantaxis_db = client["quantaxis"]
        backuplog_collection = quantaxis_db['backuplog']
        data = {'type':"full",
            'excute_t': now_time, 
            'last_rel_ts': '', 
            'folder': folder_name,
            'finish':datetime.now()}
        backuplog_collection.insert_one(data)
        print(folder_name+' full bk success')
    else:
        print(cmd)
        print('full bk fails')
        raise Exception(folder_name+' full bk fails')



def increment_backup():
    print('start increment_backup')
    client = MongoClient(dbpath)
    quantaxis_db = client["quantaxis"]
    backuplog_collection = quantaxis_db['backuplog']

    local_db = client["local"]
    oplog_collection = local_db['oplog.rs']

    now_time = datetime.now()
    folder_name = "rels" + now_time.strftime('%Y%m%d%H%M%S')

    print('prepare :'+folder_name)

    ## get last backup info, to comfirm this time period
    try:
        last = backuplog_collection.find({'type':'rels'}).sort('_id', -1).limit(1).next()
        last_record_time = last['last_rel_ts']
        start_timestamp = last_record_time.time
    except Exception as e:
        print(e)
        start_timestamp = 0

    ## get newest repSet time, to comfirm this time period
    try:
        new_record_time = oplog_collection.find().sort('ts', -1).limit(1).next()
    except Exception as e:
        raise Exception('no replSet data ||| or-> Exception:'+ str(e))

    newest_record_time = new_record_time['ts']
    stop_timestamp = newest_record_time.time

    cmd = "mongodump --port=" \
        + str(port) \
        + " -d local -c oplog.rs -q \"{ \\\"ts\\\":" \
        + "{"\
        +  "\\\"$gte\\\": { \\\"$timestamp\\\": { \\\"t\\\": " \
            + str(start_timestamp) + ", \\\"i\\\": 1 } } , " \
            + " \\\"$lte\\\": { \\\"$timestamp\\\": { \\\"t\\\": " \
            + str(stop_timestamp) + ", \\\"i\\\": 1 } } } " \
        + "}\" " \
        + "--gzip " \
        + "--out " \
        + out_path + "\\"+ folder_name +"\\"

    print(f'start subprocess:from {start_timestamp} to {stop_timestamp}')
    res = subprocess.call(cmd, shell=True)

    if res==0:
        data = {'type':"rels",
            'excute_t': now_time, 
            'last_rel_ts': newest_record_time, 
            'folder': folder_name,
            'finish':datetime.now()}
        backuplog_collection.insert_one(data)
        print(folder_name+' bk success')
    else:
        print(cmd)
        print('bk fails')
        raise Exception(folder_name+' bk fails')



if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv)<2:
        increment_backup()
    elif sys.argv[1]=='full':
        full_backup()
    elif sys.argv[1]=='increment':
        increment_backup()
    else:
        raise Exception('full or increment?')