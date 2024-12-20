
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <iostream>

const size_t address_space_size = 500 * 1024; // 50KB
const size_t element_size = 64;             // Element size 64 bytes
const size_t iterations = 100000;          // Number of iterations per run
const size_t num_repeats = 2;             // Number of times to repeat the experiment

class Timer {
  struct timespec start_time_;

public:
  Timer() {
    int rc = clock_gettime(CLOCK_MONOTONIC, &start_time_);
    assert(rc == 0);
  }

  double get_diff() {
    struct timespec end_time;
    int rc = clock_gettime(CLOCK_MONOTONIC, &end_time);
    assert(rc == 0);
    return (end_time.tv_sec - start_time_.tv_sec) * 1e9 + // Seconds to nanoseconds
           (end_time.tv_nsec - start_time_.tv_nsec);     // Nanoseconds difference
  }
};

struct LatencyData {
  size_t offset;
  double avg_latency;
};

// Write batch of data to a text file
void write_to_text_file(const std::vector<LatencyData> &latency_data, const char *filename, bool append = false) {
  std::ofstream out_file;
  if (append) {
    out_file.open(filename, std::ios::app);
  } else {
    out_file.open(filename);
  }
  assert(out_file.is_open());

  for (const auto &entry : latency_data) {
    out_file << "Offset: 0x" << std::hex << entry.offset << ", Avg Latency: " << std::dec << entry.avg_latency << " ns\n";
  }

  out_file.close();
}

void main_prog() {
  // Allocate 50 KB memory for the address space
  char *address_space = (char *)mmap(NULL, address_space_size, PROT_READ | PROT_WRITE,
                                     MAP_ANON | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
  assert(address_space != MAP_FAILED);
  memset(address_space, 0, address_space_size);

  // Initialize the address space as a pointer array
  size_t num_elements = address_space_size / element_size;
  uintptr_t *pointers = (uintptr_t *)address_space;

  // Set up the pointer chain
  pointers[0] = (uintptr_t)&pointers[1];
  for (size_t i = 1; i < num_elements; ++i) {
    pointers[i] = (uintptr_t)&pointers[0];
  }

  printf("Starting Address Space Latency Test...\n");

  // Temporary buffer to store latency data
  std::vector<LatencyData> latency_data_batch;

  for (size_t offset = 0; offset < num_elements; ++offset) {
    volatile uintptr_t *current = &pointers[offset];
    double total_latency = 0.0;

    for (size_t repeat = 0; repeat < num_repeats; ++repeat) {
      volatile uintptr_t *b;
      volatile uintptr_t *c;
      Timer timer;
      asm volatile("clflush (%0)" : : "r"(current) : "memory"); // Flush cache for current
      asm volatile("clflush (%0)" : : "r"(*current) : "memory");
      asm volatile("mfence" ::: "memory");                     // Memory fence

      timer = Timer();
      asm volatile("mfence" ::: "memory");                     // Memory fence
      b = (volatile uintptr_t *)*current; // Dereference current pointer
      asm volatile("mfence" ::: "memory");                     // Memory fence
      c = (volatile uintptr_t *)*current;
      asm volatile("mfence" ::: "memory");                     // Memory fence
      if(repeat != 0){
        total_latency += timer.get_diff(); // Accumulate nanoseconds
       }
      (void) b;
      (void) c;
    }

    double avg_latency = total_latency / (num_repeats - 1);
    latency_data_batch.push_back({offset * element_size, avg_latency});

    printf("Offset: 0x%016zx, Avg Latency: %.3f ns\n", offset * element_size, avg_latency);
  }

  // Write results to file
  write_to_text_file(latency_data_batch, "result/latency_results.txt", false);

  printf("Test completed. Results saved to 'result/latency_results.txt'.\n");

  // Cleanup
  munmap(address_space, address_space_size);
}

int main() {
  printf("Starting program...\n");
  main_prog();
  return 0;
}
