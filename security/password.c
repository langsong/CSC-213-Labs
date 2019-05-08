#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <time.h>

#define PASSWORD_LENGTH 8
#define MAX_ATTEMPTS 3

// Prompt for an input line
char* prompt(char* message) {
  printf("%s", message);
  char* line = NULL;
  size_t linecap = 0;
  if(getline(&line, &linecap, stdin) == NULL) {
    perror("getline failed");
    exit(2);
  }
  
  // Remove the newline
  line[strlen(line)-1] = '\0';
  
  return line;
}

// Generate a random password
char* generate_password(int length) {
  // Make space for a password
  char* password = malloc(sizeof(char) * (length + 1));
  
  // Fill the password in with random lowercase letters
  for(int i=0; i<length; i++) {
    password[i] = 'a' + rand() % 26;
  }
  
  // Add a null terminator
  password[length] = '\0';
  
  return password;
}

int main() {
  // Seed the random number generator
  srand(time(NULL));

  // Generate an eight-character password
  char* password = generate_password(PASSWORD_LENGTH);

  // Keep track of how many login attempts there have been
  int attempts = 0;
  
  // Keep track of whether the user has authenticated or not
  bool authenticated = false;

  // Let the user log in
  while(attempts < MAX_ATTEMPTS && !authenticated) {
    // Count this attempt
    attempts++;
 
    // Turn off echoing so the password isn't displayed
    struct termios t;
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);

    // Prompt for the password
    char* input = prompt("Password: ");

    // Restore echoing
    tcgetattr(STDIN_FILENO, &t);
    t.c_lflag |= ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
    
    // Print a newline so our next message appears on its own line
    printf("\n");

    // Check to see if the password was correct
    if(strcmp(input, password) == 0) {
      authenticated = true;
    } else{
      printf("You entered: ");
      printf(input);
      printf("\nThat is not the password. You have %d attempts remaining.\n", MAX_ATTEMPTS-attempts);
    }
    
    // Clean up after prompt
    free(input);
  }

  // Print the outcome
  if(authenticated) {
    printf("Access granted.\n");
  } else {
    printf("Access denied.\n");
  }
  
  // Free the generated password
  free(password);

  return 0;
}

