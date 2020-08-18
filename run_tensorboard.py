import os
import threading

def run_tensorboard(location = "./result/"):
    #print(f"Tensorboard Pid : {os.getpid()} & Tid : {threading.current_thread().name}")
    os.system(f"tensorboard --logdir={location}")

#print(f"Tensorboard Pid : {os.getpid()} & Tid : {threading.current_thread().name}")
print('Running tensorboard...')
# create daemon thread and run
thread = threading.Thread(target=run_tensorboard)
thread.daemon = True
thread.start()
