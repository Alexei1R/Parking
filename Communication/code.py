import mmap
import os
import time

# Create a shared memory region
shm_fd = os.open("/my_shared_memory", os.O_CREAT | os.O_RDWR)
os.write(shm_fd, b'\x00' * 1024)  # Initialize shared memory with zeros

# Memory-map the shared memory
shm = mmap.mmap(shm_fd, 1024)

# Write a message to the shared memory
message = "Hello from Python!"
shm.seek(0)
shm.write(message.encode())

# Signal that the message has been written
shm.seek(0)
shm.write(b'\x01')

# Close the shared memory file descriptor
os.close(shm_fd)

