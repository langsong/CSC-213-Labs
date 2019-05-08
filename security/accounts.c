#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PASSWORD_LENGTH 8

// Use a linked list to track accounts
typedef struct account {
  char* username;
  char* password;
  struct account* next;
} account_t;

// Head of the linked list
account_t* accounts = NULL;

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

// Add an account to the accounts list
void add_account(char* username, char* password) {
  account_t* a = malloc(sizeof(account_t));
  a->username = username;
  a->password = password;
  a->next = accounts;
  accounts = a;
}

// The user is trying to log in
void log_in() {
  // Prompt for the username
  char* username = prompt("Username: ");

  // Prompt for the password without displaying it
  char* password = getpass("Password: ");
  
  // Loop over accounts to check credentials
  account_t* current = accounts;
  while(current != NULL) {
    // Does the username match?
    if(strcmp(current->username, username) == 0) {
      // Does the password match?
      if(strcmp(current->password, password) == 0) {
        printf("Logged in.\n");
        
        // Is the logged-in user root?
        if(strcmp(current->username, "root") == 0) {
          printf("YOU ARE NOW ROOT!\n");
          exit(0);
        } else {
          printf("You seem suspicious. Good thing you're just logged in as a normal user.\n");
        }
        
        // All done
        return;
      } else {
        printf("Invalid password for user %s\n", current->username);
        return;
      }
    }
    current = current->next;
  }

  // If we hit this point we didn't find a matching username
  printf("User %s does not exist.\n", username);
}

// The user is creating an account
void create_account() {
  // Loop until the user enters a unique username
  char* username = NULL;
  bool have_username = false;
  while(!have_username) {
    // Prompt for a username
    username = prompt("What username would you like to use?\n");
    
    // Check for uniqueness
    have_username = true;
    account_t* current = accounts;
    while(current != NULL) {
      if(strcmp(username, current->username) == 0) {
        printf("That username is already in use.\n");
        have_username = false;
      }
      current = current->next;
    }
  }

  // Loop until the user has a password
  char* password = NULL;
  bool have_password = false;
  while(!have_password) {
    // Ask if we should generate a password
    char* choice = prompt("Would you like the system to generate a password for you? (y/n)");

    // Are we generating or prompting for a password?
    if(strcmp(choice, "y") == 0) {
      password = generate_password(PASSWORD_LENGTH);
      printf("Your generated password is: %s\n", password);
      have_password = true;

    } else if(strcmp(choice, "n") == 0) {
      // Prompt for a password
      password = prompt("Enter your password and hit enter.\n");
      have_password = true;

    } else {
      printf("Invalid option.\n");
    }

    // Clean up after prompt
    free(choice);
  }
  
  add_account(username, password);
  printf("Your account has been created.\n");
}

int main(int argc, char** argv) {
  printf("Account manager starting up...\n");

  printf("Seeding random number generator.\n");
  srand(time(NULL));

  printf("Creating an administrator account.\n");
  char* root_password = generate_password(PASSWORD_LENGTH);
  add_account(strdup("root"), root_password);
  free(root_password);

  printf("Welcome to the account manager!\n");

  bool running = true;
  while(running) {
    // Show options
    printf("\nWhat would you like to do?\n");
    printf("1. Log in\n");
    printf("2. Create an account\n");
    printf("3. Exit\n");
    
    // Prompt for a choice
    char* choice = prompt("\nEnter a number: ");

    // Try to convert the option to an int
    int option = atoi(choice);

    // Handle options
    if(option == 1) {
      log_in();
    } else if(option == 2) {
      create_account();
    } else if(option == 3) {
      printf("Shutting down...\n");
      running = false;
    } else {
      printf("Invalid option. Try again.\n");
    }
    
    // Clean up after prompt
    free(choice);
  }

  // Free the accounts list
  account_t* current = accounts;
  while(current != NULL) {
    account_t* next = current->next;
    free(current->username);
    free(current->password);
    free(current);
    current = next;
  }

  return 0;
}

