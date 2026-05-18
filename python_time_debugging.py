## Testing how time.time works

import time
import numpy as np



# Start while loop

i = 0
itertime = 0
starttime=time.time()

while (i<100):
    itertime = time.time()-starttime
    if itertime>5:
        print("iteration exceeded 5s")
        break
    print(f"iteration {i}: {itertime:<2.4f}s")
    time.sleep(1)
    i+=1
