#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void print_commands(char* line) {
  // Walk through pieces of the string delimited by ampersands or semicolons
  char* pos = line;
  while(1) {
    // Find the next occurrence of a semicolon or ampersand
    char* end_pos = strpbrk(pos, "&;");
    
    if(end_pos == NULL) {
      // Not found: this is the last command
      printf("Semicolon command: %s\n", pos);
      return;
    } else if(*end_pos == ';') {
      // Found a semicolon
      *end_pos = '\0';
      printf("Semicolon command: %s\n", pos);
    } else if(*end_pos == '&') {
      // Found an ampersand
      *end_pos = '\0';
      printf("Ampersand command: %s\n", pos);
    } else {
      // This should never happen, but just being thorough
      printf("Something strange happened!\n");
      return;
    }
    
    // Move the position pointer to the beginning of the next split
    pos = end_pos+1;
  }
}

int main(int argc, char** argv) {
  while(1) {
    char* line = NULL;    // Pointer that will hold the line we read in
    size_t line_length;   // Space for the length of the line
    
    // Print the shell prompt
    printf("> ");
  
    // Get a line of stdin, storing the string pointer in line and length in line_length
    if(getline(&line, &line_length, stdin) == -1) {
      if(errno == EINVAL) {
        perror("Unable to read command line");
        exit(2);
      } else {
        // Must have been end of file (ctrl+D)
        printf("\nShutting down...\n");
        exit(0);
      }
    }
    
    printf("Received input: %s", line); // No newline because line already has one
    
    print_commands(line);
    
    free(line);
  }
  
  return 0;
}
