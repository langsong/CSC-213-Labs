#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "socket.h"
#include "ui.h"

// The username for this client
const char* username;

// An array of socket file descriptors for neighbors
int* neighbors = NULL;

// The number of neighbors
size_t neighbor_count = 0;

// A lock to protect the neighbors array
pthread_mutex_t neighbors_lock = PTHREAD_MUTEX_INITIALIZER;

// A struct that holds a message header
typedef struct message_header {
  size_t username_size;
  size_t message_size;
} message_header_t;

// Add a neighbor to the list of peers
void add_neighbor(int socket_fd) {
  // Lock the neighbors data
  pthread_mutex_lock(&neighbors_lock);
  
  // Make space for a new neighbor
  neighbor_count++;
  neighbors = realloc(neighbors, sizeof(int) * neighbor_count);
  
  // Save the neighbor file descriptor
  neighbors[neighbor_count-1] = socket_fd;
  
  // Unlock
  pthread_mutex_unlock(&neighbors_lock);
}

// Remove a peer from the list given its index. The list must already be locked.
void remove_neighbor_nolock(int index) {
  memmove(&neighbors[index], &neighbors[index+1], neighbor_count - index - 1);
  neighbor_count--;
}

// Remove a peer from the list of neighbors given its socket fd
void remove_neighbor(int socket_fd) {
  pthread_mutex_lock(&neighbors_lock);
  
  for(int i=0; i<neighbor_count; i++) {
    if(neighbors[i] == socket_fd) {
      remove_neighbor_nolock(i);
      break;
    }
  }
  
  pthread_mutex_unlock(&neighbors_lock);
}

// Send a message to all neighbors. Skip the file descriptor passed in as
// skip_fd. Pass -1 to send to all.
void send_message(const char* username, const char* message, int skip_fd) {
  // Lock the neighbors array
  pthread_mutex_lock(&neighbors_lock);
  
  // Make a header to send to neighbors
  message_header_t header = {
    .username_size = strlen(username),
    .message_size = strlen(message)
  };
  
  // Loop over neighbors
  for(int i=0; i<neighbor_count; i++) {
    // Skip the specified socket file descriptor
    if(neighbors[i] == skip_fd) continue;
    
    // Send the header
    if(write(neighbors[i], &header, sizeof(message_header_t)) != sizeof(message_header_t)) {
      remove_neighbor_nolock(i);
      close(neighbors[i]);
      continue;
    }
    
    // Send the username
    if(write(neighbors[i], username, header.username_size) != header.username_size) {
      remove_neighbor_nolock(i);
      close(neighbors[i]);
      continue;
    }
    
    // Send the message
    if(write(neighbors[i], message, header.message_size) != header.message_size) {
      remove_neighbor_nolock(i);
      close(neighbors[i]);
      continue;
    }
  }
  
  // Unlock the neighbors array
  pthread_mutex_unlock(&neighbors_lock);
}

// This function is run whenever the user hits enter after typing a message
void input_callback(const char* message) {
  if(strcmp(message, ":quit") == 0 || strcmp(message, ":q") == 0) {
    ui_exit();
  } else {
    ui_display(username, message);
    send_message(username, message, -1);
  }
}

// Run this function for every peer (child or parent)
void* peer_thread_fn(void* arg) {
  // Unpack the thread argument
  int socket_fd = (int)arg;
  
  while(true) {
    // Read a message header from the socket
    message_header_t header;
    if(read(socket_fd, &header, sizeof(header)) != sizeof(header)) {
      // We must have lost our connection to this client
      remove_neighbor(socket_fd);
      close(socket_fd);
      break;
    }
    
    // Make space to hold the username and message
    char* username = malloc(header.username_size + 1);
    char* message = malloc(header.message_size + 1);
    
    // Read the username
    if(read(socket_fd, username, header.username_size) != header.username_size) {
      // We must have lost the connection
      remove_neighbor(socket_fd);
      close(socket_fd);
      break;
    }
    
    // Null-terminate the username
    username[header.username_size] = '\0';
    
    // Read the message
    if(read(socket_fd, message, header.message_size) != header.message_size) {
      // We must have lost the connection
      remove_neighbor(socket_fd);
      close(socket_fd);
      break;
    }
    
    // Null-terminate the message
    message[header.message_size] = '\0';
    
    // Display the message locally
    ui_display(username, message);
    
    // Send the message on to neighbors
    send_message(username, message, socket_fd);
    
    // Free the username and message space
    free(username);
    free(message);
  }
  
  return NULL;
}

// Run this function to accept connections from new peers
void* server_thread_fn(void* arg) {
  // Unpack the thread argument
  int socket_fd = (int)arg;
  
  while(true) {
    // Accept a new connection
    int client_socket_fd = server_socket_accept(socket_fd);
    if(client_socket_fd == -1) {
      ui_display("WARNING", "Failed to accept incoming connection.");
      break;
    }
    
    // Add the new child to the neighbors list
    add_neighbor(client_socket_fd);
    
    // Create a new thread to listen to this neighbor
    pthread_t client_thread;
    if(pthread_create(&client_thread, NULL, peer_thread_fn, (void*)(intptr_t)client_socket_fd)) {
      ui_display("WARNING", "Failed to create thread for client.");
    }
  }
  
  return NULL;
}

int main(int argc, char** argv) {
  // Make sure the arguments include a username
  if(argc != 2 && argc != 4) {
    fprintf(stderr, "Usage: %s <username> [<peer> <port number>]\n", argv[0]);
    exit(1);
  }
  
  // Save the username in a global
  username = argv[1];
  
  // Set up server socket
  unsigned short port;
  int server_socket_fd = server_socket_open(&port);
  if(server_socket_fd == -1) {
    perror("Failed to create server socket");
    exit(2);
  }
  
  // Start listening
  if(listen(server_socket_fd, 8)) {
    perror("Failed to listen on server socket");
    exit(2);
  }
  
  // Create a thread to accept new connections
  pthread_t server_thread;
  if(pthread_create(&server_thread, NULL, server_thread_fn, (void*)(intptr_t)server_socket_fd)) {
    perror("Failed to create server thread");
    exit(2);
  }
  
  // If a peer was specified, try to connect
  if(argc == 4) {
    // Unpack arguments
    char* peer_hostname = argv[2];
    unsigned short peer_port = atoi(argv[3]);
    
    // Connect
    int peer_socket_fd = socket_connect(peer_hostname, peer_port);
    if(peer_socket_fd == -1) {
      perror("Failed to connect to specified peer");
      exit(2);
    }
    
    // Add the peer to the neighbors list
    add_neighbor(peer_socket_fd);
    
    // Create a thread to talk to the peer
    pthread_t peer_thread;
    if(pthread_create(&peer_thread, NULL, peer_thread_fn, (void*)(intptr_t)peer_socket_fd)) {
      perror("Failed to create peer thread");
      exit(2);
    }
  }
  
  // Set up the user interface. The input_callback function will be called
  // each time the user hits enter to send a message.
  ui_init(input_callback);
  
  // Show the server state in the message display
  char server_message[256];
  snprintf(server_message, 256, "Server is running on port %u.", port);
  ui_display("INFO", server_message);
  
  // Run the UI loop
  ui_run();
  
  return 0;
}
