#ifndef LIST_H
#define LIST_H

#include <stdbool.h>
#include <pthread.h>

// This makes the header file work for both C and C++
#ifdef __cplusplus
extern "C" {
#endif

char *strdup(const char *s);
  
typedef struct node {
  struct node* next;
  char* key;
  int value;
} node_t;
  
typedef struct my_list {
  node_t* head;
  pthread_mutex_t m;
} my_list_t;

// Initialize a list
void list_init(my_list_t* list);

// Destroy a list
void list_destroy(my_list_t* list);

// Push an element onto a list
void list_insert(my_list_t* list, const char* key, int val);

// Pop an element off of a list
int list_remove(my_list_t* list, const char* key);

// Get the value of the corresponding key
int list_lookup(my_list_t* list, const char* key);

// This makes the header file work for both C and C++
#ifdef __cplusplus
}
#endif

#endif
