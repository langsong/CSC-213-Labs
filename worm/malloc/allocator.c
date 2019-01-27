#define _GNU_SOURCE

#include <assert.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

// The minimum size returned by malloc
#define MIN_MALLOC_SIZE 16

// Round a value x up to the next multiple of y
#define ROUND_UP(x,y) ((x) % (y) == 0 ? (x) : (x) + ((y) - (x) % (y)))

// Round a value x down to the next multiple of y
#define ROUND_DOWN(x,y) ((x) % (y) == 0 ? (x) : (x) - ((x) % (y)))

// The size of a single page of memory, in bytes
#define PAGE_SIZE 0x1000

// USE ONLY IN CASE OF EMERGENCY
bool in_malloc = false;           // Set whenever we are inside malloc.
bool use_emergency_block = false; // If set, use the emergency space for allocations
char emergency_block[1024];       // Emergency space for allocating to print errors

// Declare this function so we can use it in xxfree
size_t xxmalloc_usable_size(void* ptr);

typedef struct __attribute__((packed))free_list {
  struct free_list* next;
} free_list_t;

typedef struct __attribute__((packed)) header {
  size_t magic;
  size_t size;
  struct header* next;
  free_list_t* list;
} header_t;

typedef struct __attribute__((packed)) large_header {
  size_t magic;
  size_t size;
} large_header_t;

header_t* array_blocklist[8] = { NULL };

void* find_header(void* ptr) {
  return (void *) ROUND_DOWN( (intptr_t)ptr, PAGE_SIZE );
}

size_t round_to_power_two(size_t num) {
  size_t tz = __builtin_ctzll(num); // trailing zero
  size_t lz = __builtin_clzll(num); // leading zero
  if(64 - tz - lz == 1) {
    return num;
  } else {
    return 1 << (64 - lz);
  }
}

header_t* setup_new_block(size_t block_size, size_t size) {
  // Request memory from the operating system in page-sized chunks
  void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

  // Check for errors
  if(p == MAP_FAILED) {
    use_emergency_block = true;
    perror("mmap");
    exit(2);
  }

  header_t* cur_header = (header_t*) p;
  bool is_16 = (block_size == 16);

  // Setting up the header
  cur_header->size = block_size;
  cur_header->next = NULL;
  cur_header->magic = 0xD00FCA75; // Magic number for normal block
  intptr_t pi = (intptr_t)p;
  p = (void*)(pi + ROUND_UP(sizeof(header_t), block_size)); // Padding the header

  free_list_t* freelist = (free_list_t*)p;
  cur_header->list = freelist;

  // Setting up the freelist
  pi = (intptr_t)p + block_size;
  for(size_t i = is_16 ? 3 : 2; i < size / block_size; i++) { // starting from either 1 or 2 depends on if it's a 16 byte object
    freelist->next = (void*)pi;
    freelist = freelist->next;
    pi += block_size;
  }
  freelist->next = NULL;
  return cur_header;
}

/**
 * Allocate space on the heap.
 * \param size  The minimium number of bytes that must be allocated
 * \returns     A pointer to the beginning of the allocated space.
 *              This function may return NULL when an error occurs.
 */
void* xxmalloc(size_t size) {
  // Before we try to allocate anything, check if we are trying to print an error or if
  // malloc called itself. This doesn't always work, but sometimes it helps.
  if(use_emergency_block) {
    return emergency_block;
  } else if(in_malloc) {
    use_emergency_block = true;
    // printf("here: %lu\n", size);
    puts("ERROR! Nested call to malloc. Aborting.\n");
    exit(2);
  }

  // If we call malloc again while this is true, bad things will happen.
  in_malloc = true;
  size_t block_size = round_to_power_two(size);
  block_size = block_size > 16 ? block_size : 16;

  // Find the corresponding bucket
  int index = __builtin_ctzll(block_size) - 4;

  // Round the size up to the next multiple of the page size
  size = ROUND_UP(size, PAGE_SIZE);

  if(block_size > 2048) { // We have a big block
    // Request memory from the operating system in page-sized chunks
    void* p = mmap(NULL, size+PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    // Check for errors
    if(p == MAP_FAILED) {
      use_emergency_block = true;
      perror("mmap");
      exit(2);
    }  
    large_header_t* cur_large_header = (large_header_t*) p;
    cur_large_header->size = size;
    cur_large_header->magic = 0xF00DFACE; // Big block magic number    
    
    in_malloc = false;
    return (void*)((intptr_t)p + PAGE_SIZE);
  }

  header_t* cur_header = array_blocklist[index];

  // Find right header or create one
  if(cur_header == NULL) { // The block list is not existed yet
    array_blocklist[index] = setup_new_block(block_size, size);
    cur_header = array_blocklist[index];
  } else {
    while(cur_header->list == NULL) {
      if(cur_header->next == NULL) { // There is no available memory, we have to mmap a new block
        cur_header->next = setup_new_block(block_size, size);
        cur_header = cur_header->next;
        break;
      } else {
        cur_header = cur_header->next; // Advance to the next block
      }
    }
  }
  free_list_t* cur_freelist = cur_header->list;
  cur_header->list = cur_freelist->next;

  // Done with malloc, so clear this flag
  in_malloc = false;

  return cur_freelist;
}

/**
 * Free space occupied by a heap object.
 * \param ptr   A pointer somewhere inside the object that is being freed
 */
void xxfree(void* ptr) {
  if(ptr == NULL) {
    return;
  }
  size_t block_size = xxmalloc_usable_size(ptr);
  if(block_size > 2048) { // Big block
    large_header_t* large_header = (large_header_t*)find_header((void*)((intptr_t)ptr - 1));
    while(large_header->magic != 0xF00DFACE) { // Find the header for the large object
      large_header = (large_header_t*)find_header((void*)((intptr_t)large_header - 1)); // Round down to the next page to find the header
    }
    munmap((void*)((intptr_t)large_header + PAGE_SIZE), block_size);
  } else { // Normal block
    free_list_t* start = (free_list_t*)ROUND_DOWN((intptr_t)ptr, block_size);
    int index = __builtin_ctzll(block_size) - 4; // Find bucket
    header_t* cur_header = array_blocklist[index];
    free_list_t* list = cur_header->list;
    start->next = list; 
    cur_header->list = start;
  }
}

/**
 * Get the available size of an allocated object
 * \param ptr   A pointer somewhere inside the allocated object
 * \returns     The number of bytes available for use in this object
 */
size_t xxmalloc_usable_size(void* ptr) {
  // We aren't tracking the size of allocated objects yet, so all we know is that it's at least PAGE_SIZE bytes.
  header_t* cur_header = (header_t*)find_header(ptr);
  if(cur_header->magic == 0xD00FCA75) { // It's a normal block
    return cur_header->size;
  } else { // It's a large block
    large_header_t* large_header = (large_header_t*)find_header((void*)((intptr_t)ptr - 1));
    while(large_header->magic != 0xF00DFACE) { // Find the header for the large object
      large_header = (large_header_t*)find_header((void*)((intptr_t)large_header - 1));
    }
    return large_header->size;
  }
}
