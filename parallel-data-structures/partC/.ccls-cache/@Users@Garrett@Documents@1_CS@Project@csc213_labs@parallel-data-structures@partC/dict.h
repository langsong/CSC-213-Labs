#ifndef DICT_H
#define DICT_H

#include <stdbool.h>
#include "list.h"

// This makes the header file work for both C and C++
#ifdef __cplusplus
extern "C" {
#endif

#define BUCKETS 100
  
typedef struct my_dict {
  my_list_t lists[BUCKETS];
} my_dict_t;

// Hash Function
unsigned long hash(const char* str);
  
// Initialize a dictionary
void dict_init(my_dict_t* dict);

// Destroy a dictionary
void dict_destroy(my_dict_t* dict);

// Set a value in a dictionary
void dict_set(my_dict_t* dict, const char* key, int value);

// Check if a dictionary contains a key
bool dict_contains(my_dict_t* dict, const char* key);

// Get a value in a dictionary
int dict_get(my_dict_t* dict, const char* key);

// Remove a value from a dictionary
void dict_remove(my_dict_t* dict, const char* key);

// This makes the header file work for both C and C++
#ifdef __cplusplus
}
#endif

#endif
