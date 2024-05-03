#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>

int main() {
    // Open the shared memory region
    int shm_fd = shm_open("/my_shared_memory", O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Error opening shared memory" << std::endl;
        return 1;
    }

    // Memory-map the shared memory
    void* ptr = mmap(0, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "Error mapping shared memory" << std::endl;
        return 1;
    }

    // Open the semaphore
    sem_t* semaphore = sem_open("/my_semaphore", O_CREAT, 0666, 0);
    if (semaphore == SEM_FAILED) {
        std::cerr << "Error opening semaphore" << std::endl;
        return 1;
    }

    // Wait for the sender to signal that the message is ready
    sem_wait(semaphore);

    // Read the message from shared memory
    std::string message(static_cast<const char*>(ptr));

    // Print the received message
    std::cout << "Received message: " << message << std::endl;

    // Close and unlink shared memory and semaphore
    close(shm_fd);
    shm_unlink("/my_shared_memory");
    sem_close(semaphore);
    sem_unlink("/my_semaphore");

    return 0;
}

