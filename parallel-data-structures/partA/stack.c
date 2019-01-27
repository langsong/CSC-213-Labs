#include "stack.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

// Initialize a stack
void stack_init(my_stack_t* stack) {
  stack->head = NULL;
  stack->size = 0;
  pthread_mutex_init(&stack->m, NULL);
}

// Destroy a stack
void stack_destroy(my_stack_t* stack) {
  node_t* cur = stack->head;
  node_t* next = NULL;
  while(cur != NULL) {
    next = cur->next;
    free(cur);
    cur = next;
  }
}

// Push an element onto a stack
void stack_push(my_stack_t* stack, int element) {
  pthread_mutex_lock(&stack->m);
  node_t* temp = malloc(sizeof(node_t));
  if(temp == NULL) {
    fprintf(stderr, "Malloc failed!\n");
    stack_destroy(stack);
    exit(1);
  }
  temp->data = element;
  temp->next = stack->head;
  stack->head = temp;
  stack->size++;
  pthread_mutex_unlock(&stack->m);
}

// Check if a stack is empty
bool stack_empty(my_stack_t* stack) {
  return stack->size <= 0;
}

// Pop an element off of a stack
int stack_pop(my_stack_t* stack) {
  if(stack_empty(stack)) {
    return -1;
  }
  pthread_mutex_lock(&stack->m);
  node_t* cur = stack->head;
  int data = cur->data;
  stack->head = cur->next;
  stack->size--;
  pthread_mutex_unlock(&stack->m);
  return data;
}
