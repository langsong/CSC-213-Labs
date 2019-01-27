#include "queue.h"

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

// Initialize a new queue
void queue_init(my_queue_t* queue) {
  node_t* temp = malloc(sizeof(node_t));
  if(temp == NULL) {
    fprintf(stderr, "Malloc failed!\n");
    exit(1);
  }
  temp->next = NULL;
  queue->head = queue->tail = temp;
  pthread_mutex_init(&queue->headlock, NULL);
  pthread_mutex_init(&queue->taillock, NULL);
}

// Destroy a queue
void queue_destroy(my_queue_t* queue) {
  node_t* cur = queue->head;
  node_t* next = NULL;
  while(cur != NULL) {
    next = cur->next;
    free(cur);
    cur = next;
  }
}

// Put an element at the end of a queue
void queue_put(my_queue_t* queue, int element) {
  node_t* temp = malloc(sizeof(node_t));
  if(temp == NULL) {
    fprintf(stderr, "Malloc failed!\n");
    exit(1);
  }
  temp->data = element;
  temp->next = NULL;

  pthread_mutex_lock(&queue->taillock);
  queue->tail->next = temp;
  queue->tail = temp;
  pthread_mutex_unlock(&queue->taillock);
}

// Check if a queue is empty
bool queue_empty(my_queue_t* queue) {
  return queue->head == queue->tail;
}

// Take an element off the front of a queue
int queue_take(my_queue_t* queue) {
  pthread_mutex_lock(&queue->headlock);
  node_t* temp = queue->head;
  node_t* new_head = temp->next;
  if(new_head == NULL) {
    pthread_mutex_unlock(&queue->headlock);
    return -1;
  }
  int data = new_head->data;
  queue->head = new_head;
  pthread_mutex_unlock(&queue->headlock);
  free(temp);
  return data;
}
