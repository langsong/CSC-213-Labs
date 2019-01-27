#include "dict.h"

#include <stdlib.h>

// Reference: https://stackoverflow.com/questions/7666509/
unsigned long hash(const char* str) {
  unsigned long hash = 5381;
  int c;

  while((c = *str++) != 0)
    hash = ((hash << 5) + hash) + c;

  return hash;
}

// Initialize a dictionary
void dict_init(my_dict_t* dict) {
  for(int i = 0; i < BUCKETS; i++) {
    list_init(&dict->lists[i]);
  }
}

// Destroy a dictionary
void dict_destroy(my_dict_t* dict) {
  for(int i = 0; i < BUCKETS; i++) {
    list_destroy(&dict->lists[i]);
  }
}

// Set a value in a dictionary
void dict_set(my_dict_t* dict, const char* key, int value) {
  int bucket = hash(key) % BUCKETS;
  list_insert(&dict->lists[bucket], key, value);
}

// Check if a dictionary contains a key
bool dict_contains(my_dict_t* dict, const char* key) {
  int bucket = hash(key) % BUCKETS;
  if (list_lookup(&dict->lists[bucket], key) != -1) {
    return true;
  } else {
    return false;
  }
}

// Get a value in a dictionary
int dict_get(my_dict_t* dict, const char* key) {
  int bucket = hash(key) % BUCKETS;
  return list_lookup(&dict->lists[bucket], key);
}

// Remove a value from a dictionary
void dict_remove(my_dict_t* dict, const char* key) {
  int bucket = hash(key) % BUCKETS;
  list_remove(&dict->lists[bucket], key);
}
